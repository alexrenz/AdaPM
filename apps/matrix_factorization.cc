#include "utils.h"
#include "ps/ps.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <thread>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/timer/timer.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>
#include <unistd.h>
#include <bitset>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "mf/update.h"
#include "mf/data.h"
#include "mf/io.h"

using namespace ps;
using namespace std;

typedef double ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;

typedef Eigen::VectorXd vect;
typedef Eigen::SparseMatrix<ValT,Eigen::RowMajor> SpMatrixRM;
typedef Eigen::SparseMatrix<ValT,Eigen::ColMajor> SpMatrixCM;
typedef Eigen::Matrix<ValT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DeMatrixRM;
typedef Eigen::Matrix<ValT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DeMatrixCM;

enum class Alg {dsgd, columnwise, plain_sgd};
std::ostream& operator<<(std::ostream &o, const Alg& rsm) {
  switch(rsm) {
  case Alg::dsgd: return o << "dsgd";
  case Alg::columnwise: return o << "columnwise";
  case Alg::plain_sgd: return o << "plain_sgd";
  default: return o << "unkown";
  }
}
std::istream& operator>>(std::istream& in, Alg& rsm) {
  std::string token; in >> token;
  if (token == "dsgd") rsm = Alg::dsgd;
  else if (token == "columnwise") rsm = Alg::columnwise;
  else if (token == "plain_sgd") rsm = Alg::plain_sgd;
  else { CHECK(false) << "Fatal! Unknown MF algorithm " << token; }
  return in;
}

// Matrix factorization config
string dataset;
uint mf_rank;
int epochs;
double initial_step_size;
double lambda;
double increase_step_factor;
double decrease_step_factor;
bool use_wor_block_schedule;
bool use_wor_point_schedule;
bool compute_loss;
bool bold_driver;

// System config
uint num_workers;
int  num_threads;
int  num_cols;
int  num_rows;
bool shared_memory;
Alg algorithm;
bool disable_localize_blocks;
bool disable_localize_left;
size_t prelocalize_steps;
size_t prelocalize_groupsize;
int init_parameters;
bool enforce_random_keys;
unsigned long early_stop;
bool sync_push;
size_t replicate;
long max_runtime;
bool prevent_full_model_pull;

// set by system
Key first_col_key;
Key num_keys;
std::vector<Key> key_assignment;
unsigned model_seed;

// Shared data
std::vector<Key> full_h_keys;
std::vector<ValT> full_h;

inline Key row_key(size_t i) {
  return enforce_random_keys ? key_assignment[i] : i;
}
inline Key col_key(size_t j) {
  return enforce_random_keys ? key_assignment[first_col_key+j] : first_col_key+j;
}

#include "mf/loss.h"

// Calculate loss on training and test data
double calculate_loss(const int epoch, std::vector<Key>& local_w_keys,
                      mf::DataPart& train_data, mf::DataPart& test_data,
                      WorkerT& kv, const int worker_id, const int num_workers, const int customer_id) {

  // key and vector for aggregation (in PS)
  std::vector<Key> loss_key { num_keys-1 };
  std::vector<ValT> agg_vec (mf_rank);
  assert(mf_rank >= 2); // need two elements to aggregate train and test loss


  // reset aggregation
  if (worker_id == 0) {
    kv.Wait(kv.Pull(loss_key, &agg_vec));
    for(size_t i=0; i!=agg_vec.size(); ++i) {
      agg_vec[i] = -agg_vec[i];
    }
    kv.Wait(kv.Push(loss_key, agg_vec));
  }
  kv.Barrier();

  // wait for sync
  kv.WaitReplicaSync();

  // calculate local loss
  double local_train_loss = 0;
  double local_test_loss = 0;
  if (!prevent_full_model_pull) {
    // pull local part of W and H (once per process)
    if (worker_id == 0) ADLOG("Pull model for loss calc");
    std::vector<ValT> local_w (local_w_keys.size() * mf_rank);
    int t_pullw = kv.Pull(local_w_keys, &local_w); // fetch local part of w
    if (customer_id == 1) { // fetch full h once per process
      full_h.resize(num_cols * mf_rank);
      kv.Wait(kv.Pull(full_h_keys, &full_h));
    }
    kv.Wait(t_pullw);
    kv.Barrier();

    // calculate loss (with a copy of the model)
    local_train_loss = loss_Nzsl_L2(train_data, local_w, full_h, lambda, mf_rank, worker_id);
    local_test_loss =  loss_Nzsl_L2(test_data,  local_w, full_h, 0.0,    mf_rank, worker_id);
  } else {
    // calculate loss (pulling necessary factors on demand)
    if (worker_id == 0) ADLOG("No full model pull for eval. Pull factors on demand.");
    local_train_loss = loss_Nzsl_L2_pull(train_data, kv, local_w_keys, lambda, mf_rank, worker_id);
    local_test_loss =  loss_Nzsl_L2_pull(test_data,  kv, local_w_keys, 0.0,    mf_rank, worker_id);
  }
  // ADLOG("Local training loss (worker " << worker_id << "): " << local_train_loss);
  // ADLOG("Local test loss (worker " << worker_id << "): " << local_test_loss);

  // aggregate local losses to global loss (using the PS)
  agg_vec[0] = local_train_loss;
  agg_vec[1] = local_test_loss;
  kv.Wait(kv.Push(loss_key, agg_vec));
  kv.Barrier();
  kv.Wait(kv.Pull(loss_key, &agg_vec));

  auto train_loss = agg_vec[0] / train_data.total_nnz();
  auto test_loss = agg_vec[1] / test_data.total_nnz();

  if (worker_id == 0) {
    ADLOG("(Ep." << epoch << ") Train loss: " << train_loss);
    ADLOG("(Ep." << epoch << ") Test loss:  " << test_loss);
  }

  return train_loss;
}


// determine which parameters to replicate:
// we replicate the factors of the N most frequent columns
std::vector<Key> determine_hotspot_keys (size_t N_hotspots) {
  std::vector<Key> hotspot_keys {};
  if (N_hotspots == 0) return hotspot_keys;

  hotspot_keys.reserve(N_hotspots);

  // stats
  long freq_replicated = 0;
  long freq_total = 0;

  // read which relations have not been partitioned
  ifstream colfreqs_ifs(dataset+"train.mmc.bin.colfreqs.csv", ios::in);
  if (!colfreqs_ifs.fail()) {
    long freq;
    long j;
    string line;

    // read header line
    getline(colfreqs_ifs, line);

    while (getline(colfreqs_ifs, line)) {
      stringstream ss(line);
      ss >> freq >> j;
      freq_total += freq;
      if (hotspot_keys.size() < N_hotspots) {
        hotspot_keys.push_back(col_key(j));
        freq_replicated += freq;
      }
    }
    colfreqs_ifs.close();
  } else {
    ADLOG("Did not find colfreqs file. Don't know which columns to replicate");
    abort();
  }
  ADLOG("Replicate " << hotspot_keys.size() << " keys. This should cover " << std::setprecision(5) << 100.0*freq_replicated/freq_total << "% of column accesses (and " << 100.0*freq_replicated/(freq_total*2) << "% of all accesses)");
  // ADLOG("Replicate columns " << hotspot_keys);
  return hotspot_keys;
}

void RunWorker(int customer_id, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server);

  boost::random::mt19937 rng(static_cast<unsigned int>(std::time(0)));
  std::unordered_map<std::string, util::Stopwatch> sw {};
  util::Trace trace {"./measurements/mf/mf_trace.csv"};

  UpdateNsqlL2 update_fun {mf_rank, initial_step_size, lambda};
  int worker_id = ps::MyRank()*num_threads+customer_id-1; // a unique id for this worker thread
  Eigen::MatrixXi block_schedule = mf::initBlockSchedule(num_workers);


  /* Load data, factors, and allocate memory */

  // Load train data
  sw["read_data"].start();
  mf::DataPart data = read_sparse_matrix_part(dataset + string("train.mmc"), true, num_workers, worker_id, num_rows, num_cols, customer_id, algorithm == Alg::dsgd, kv);
  mf::DataPart data_test = read_sparse_matrix_part(dataset + string("test.mmc"), true, num_workers, worker_id, num_rows, num_cols, customer_id, algorithm == Alg::dsgd, kv);
  sw["read_data"].stop();
  ALOG("Finished reading train data in worker " << worker_id << " (" << sw["read_data"] << ")");

  // allocate memory for factors in training loop
  std::vector<Key> factors_keys ( 2 );
  std::vector<ValT> factors ( factors_keys.size() * mf_rank );
  std::vector<ValT> factors_update ( factors.size() );

  // allocate memory for factors in loss computation (once per process)
  if (customer_id == 1) {
    full_h_keys.resize(num_cols);
    for(size_t j=0; j!=full_h_keys.size(); ++j) {
      full_h_keys[j] = col_key(j);
    }
  }

  std::vector<Key> local_h_keys ( data.num_cols_per_block() );
  std::vector<Key> local_w_keys ( data.num_rows_per_block() );
  for(size_t z=0; z!=data.num_rows_per_block(); ++z) {
    local_w_keys[z] = row_key(data.num_rows_per_block() * worker_id + z);
  }

  boost::random::uniform_real_distribution<> factor_generator(0,1);

  // push factors into the parameter servers
  kv.BeginSetup();
  if (worker_id == 0 && init_parameters > 0) {
    // read and send H
    sw["read_factors"].start();
    std::vector<ValT> init_h {};
    if (init_parameters == 1) {
      // read factors from file
      init_h = read_dense_matrix<std::vector<ValT>>(dataset + "H.mma", false);
    } else if (init_parameters == 2) {
      // init model randomly
      ALOG("Init H randomly (seed " << model_seed << ")");
      init_h.resize(num_cols * mf_rank);
      boost::random::mt19937 rng_H (model_seed+133273);
      for (size_t z=0; z!=init_h.size(); ++z) {
        init_h[z] = factor_generator(rng_H);
      }
    }
    kv.Wait(kv.StaggeredPush(full_h_keys, init_h));
    sw["read_factors"].stop();
    ALOG("Initialized H (" << sw["read_factors"] << "): " << init_h[0] << " " << init_h[1] << " " << init_h[2] << " ..");
  }

  if (init_parameters > 0 && static_cast<uint>(worker_id) == num_workers-1) {
    // read and send W
    sw["read_factors"].start();
    ValT first, second;
    if (init_parameters == 1) {
      // read factors from files
      vector<Key> full_w_keys (num_rows);
      for(size_t i = 0; i!=full_w_keys.size(); ++i) full_w_keys[i] = row_key(i);
      std::vector<ValT> full_w;
      full_w = read_dense_matrix<std::vector<ValT>>(dataset + "W.mma", true);
      kv.Wait(kv.StaggeredPush(full_w_keys, full_w));
      first = full_w[0];
      second = full_w[1];
    } else if (init_parameters == 2) {
      // init model randomly
      ALOG("Init W randomly (seed " << model_seed << ")");
      boost::random::mt19937 rng_W (model_seed);

      // use (more complicated) batched implementation so we can run large models on a single machine
      size_t batch_size = 10000;
      vector<Key> partial_w_keys {};
      std::vector<ValT> partial_w {};
      vector<int> tss {};
      size_t gen = 0;
      for (int i=0; i!=num_rows; ++i) {
        partial_w_keys.push_back(row_key(i));

        // generate
        for (size_t z=0; z!=mf_rank; ++z) {
          auto r = factor_generator(rng_W);
          if (gen == 0) first = r;
          if (gen == 1) second = r;
          ++gen;
          partial_w.push_back(r);
        }

        // push
        if (i % batch_size == batch_size-1) {
          tss.push_back(kv.Push(partial_w_keys, partial_w));
          partial_w_keys.resize(0);
          partial_w.resize(0);
        }
      }
      tss.push_back(kv.Push(partial_w_keys, partial_w));
      for (auto ts : tss) kv.Wait(ts);
    }
    sw["read_factors"].stop();
    ALOG("Initialized W (" << sw["read_factors"] << "): " << first << " " << second << " ..");
  }
  kv.EndSetup(); // waits for replica sync and does barrier

  // Localize local part of W (left matrix)
  if (!disable_localize_left) {
    kv.Wait(kv.Localize(local_w_keys));
  }

  // prep for column-wise access
  std::vector<size_t> my_columns {};
  if (algorithm == Alg::columnwise) {
    for (int j=0; j!=num_cols; ++j) {
      if (data.block_has_nnz(j)) {
        my_columns.push_back(j);
      }
    }

    srand(worker_id ^ 13 * 17); // make sure workers have unique column permutations
    if (use_wor_block_schedule) std::random_shuffle(my_columns.begin(), my_columns.end());
  }

  // Compute initial loss
  ValT previous_train_loss=0;
  if (compute_loss) {
    auto train_loss = calculate_loss(0, local_w_keys, data, data_test,
                                     kv, worker_id, num_workers, customer_id);
    previous_train_loss = train_loss;
  }

  kv.Barrier(); // make sure all workers start training at the same time

  /* Training loop */
  for(int epoch = 1; epoch != epochs+1; ++epoch) {
    if (worker_id == 0) ADLOG("(Ep." << epoch << ") Starting epoch " << epoch);
    sw["epoch.worker"].start();
    sw["epoch"].start();
    sw["runtime"].resume();

    // create WOR block schedule for this epoch, if desired (WOR = random, without replacement)
    if (use_wor_block_schedule) mf::newWorSchedule(block_schedule, epoch, rng);

    long updates = 0;

    if (algorithm == Alg::dsgd) { // use DSGD to create locality in parameter accesses
      for(uint subepoch = 0; subepoch != num_workers; ++subepoch) {
        if (worker_id == 0) ADLOG("Subepoch " << epoch << "." << subepoch);

        // get the block this workers works on in this epoch
        sw["comm"].resume();
        int h_block = block_schedule(subepoch, worker_id);
        if (!disable_localize_blocks) {
          for (size_t z = 0; z!=local_h_keys.size(); ++z) {
            local_h_keys[z] = col_key(h_block * data.num_cols_per_block() + z);
          }
          kv.Localize(local_h_keys);
        }
        sw["comm"].stop();

        // permute training points (WOR training schedule)
        sw["perm"].resume();
        if (use_wor_point_schedule) data.permuteBlock(h_block);
        sw["perm"].stop();

        // train on this block

        // iterate over the nonzeros in this block
        sw["comp"].resume();
        for (unsigned long z=data.block_start(h_block); z!=data.block_end(h_block); ++z) {
          const mf::DataPoint& dp = data.data()[data.permutation()[z]];

          // get factors
          factors_keys[0] = row_key(dp.i);
          factors_keys[1] = col_key(dp.j);
          kv.Wait(kv.Pull(factors_keys, &factors));

          // run update
          update_fun(dp.i, dp.j, factors, factors_update, dp.x, data.row_nnz(dp.i), data.col_nnz(dp.j));
          ++updates;

          // send factor updates
          auto ts = kv.Push(factors_keys, factors_update);
          if (sync_push) kv.Wait(ts);
        }

        sw["comp"].stop();

        // wait for all workers to finish the subepoch
        sw["barrier"].resume();
        kv.Barrier();
        sw["barrier"].stop();

        // ADLOG("Subepoch " << epoch << "." << subepoch << " took " << sw["subepoch"]);
      }
    } else if (algorithm == Alg::columnwise) { // columnwise
      // permute columns
      sw["perm"].resume();
      ADLOG("Permute columns");
      if(use_wor_block_schedule) std::random_shuffle(my_columns.begin(), my_columns.end());
      sw["perm"].stop();

      // iterate over the nonzeros that are partitioned to this worker
      sw["comp"].resume();
      long col_future=0;
      long z_total=0;
      long block_start_future=0;
      for (unsigned long col=0; col!=my_columns.size(); ++col) {
        auto j = my_columns[col];

        // permute data points of this column
        if (use_wor_point_schedule) data.permuteBlock(j);

        for (unsigned long z=data.block_start(j); z!=data.block_end(j); ++z, ++z_total) {
          const mf::DataPoint& dp = data.data()[data.permutation()[z]];

          // Pre-localize column parameters
          if (prelocalize_steps != 0 && z_total % prelocalize_groupsize == 0) {
            vector<Key> localize;
            localize.reserve(prelocalize_groupsize);
            long localize_until = z_total + (prelocalize_steps+1)*prelocalize_groupsize; // +1 gets us to the block end
            while (col_future < static_cast<long>(my_columns.size()) && block_start_future < localize_until) {
              localize.push_back(col_key(my_columns[col_future]));
              block_start_future += data.block_size(my_columns[col_future]);
              ++col_future;
            }
            kv.Localize(localize);
          }

          // get factors
          factors_keys[0] = row_key(dp.i);
          factors_keys[1] = col_key(dp.j);
          kv.Wait(kv.Pull(factors_keys, &factors));

          // run update
          update_fun(dp.i, dp.j, factors, factors_update, dp.x, data.row_nnz(dp.i), data.col_nnz(dp.j));
          ++updates;

          // send factor updates
          auto ts = kv.Push(factors_keys, factors_update);
          if (sync_push) kv.Wait(ts);

          // early stopping (for debugging)
          if(early_stop != 0 && z == early_stop) {
            ALOG("Worker " << worker_id << " stops the epoch early after " << early_stop << " of " << data.num_nnz() << " data points");
            goto endepoch;
          }
        }
      }
    endepoch:
      sw["comp"].stop();
      sw["epoch.worker"].stop();
      ALOG("Worker " << worker_id << " is through its " << data.num_nnz() << " data points (" << sw["epoch.worker"] << ")");

      // wait for all workers to finish the epoch
      sw["barrier"].resume();
      kv.Barrier();
      sw["barrier"].stop();

    } else { // plain SGD: access the data points in fully random order (i.e., no blocks and no columnwise access)

      // permute training data points (WOR training schedule)
      sw["perm"].resume();
      if (worker_id == 0) ADLOG("Permute ...");
      if (use_wor_point_schedule) data.permuteData();
      sw["perm"].stop();

      // iterate over the nonzeros that are partitioned to this worker
      sw["comp"].resume();
      if (worker_id == 0) ADLOG("Compute ...");
      unsigned long z_future=0;
      for (unsigned long z=0; z!=data.num_nnz(); ++z) {
        const mf::DataPoint& dp = data.data()[data.permutation()[z]];

        // early stopping (for debugging)
        if(early_stop != 0 && z == early_stop) {
          ALOG("Worker " << worker_id << " stops the epoch early after " << early_stop << " of " << data.num_nnz() << " data points");
          break;
        }

        // Pre-localize parameters
        if (prelocalize_steps != 0 && z % prelocalize_groupsize == 0) {
          vector<Key> localize;
          localize.reserve(prelocalize_groupsize);
          while (z_future < z+static_cast<int>((prelocalize_steps+1)*prelocalize_groupsize) && z_future < data.num_nnz()) {
            const mf::DataPoint& dp_f = data.data()[data.permutation()[z_future]];
            localize.push_back(col_key(dp_f.j));
            ++z_future;
          }
          kv.Localize(localize);
        }

        // get factors
        factors_keys[0] = row_key(dp.i);
        factors_keys[1] = col_key(dp.j);
        kv.Wait(kv.Pull(factors_keys, &factors));

        // run update
        update_fun(dp.i, dp.j, factors, factors_update, dp.x, data.row_nnz(dp.i), data.col_nnz(dp.j));
        ++updates;

        // send factor updates
        auto ts = kv.Push(factors_keys, factors_update);
        if (sync_push) kv.Wait(ts);
      }
      sw["comp"].stop();
      sw["epoch.worker"].stop();
      ALOG("Worker " << worker_id << " is through its " << data.num_nnz() << " data points (" << sw["epoch.worker"] << ")");

      // wait for all workers to finish the epoch
      sw["barrier"].resume();
      kv.Barrier();
      sw["barrier"].stop();

    }
    sw["epoch"].stop();
    sw["runtime"].stop();
    ADLOG("(Ep." << epoch << ") Worker " << worker_id << " finished epoch " << epoch <<" (" << sw["epoch"] << ": " << sw["comm"] << " comm, "  << sw["comp"] << " comp, " << sw["barrier"] << " barrier, updates: " << updates <<")");
    if (worker_id == 0) {
      ALOG("All workers finished epoch " << epoch << " (epoch: " << sw["epoch"] << ", total: " << sw["runtime"] << ").");
    }
    sw["comm"].reset(); sw["comp"].reset(); sw["perm"].reset(); sw["barrier"].reset();

    if (compute_loss) {
      auto train_loss = calculate_loss(epoch, local_w_keys, data, data_test,
                                       kv, worker_id, num_workers, customer_id);

      // bold driver: adapt step size
      if (bold_driver) {
        if (train_loss < previous_train_loss) {
          update_fun.update_step_size(increase_step_factor); // good epoch. increase step size slightly
          if (worker_id == 0) {
            ADLOG("Bold driver: Increase step size to " << update_fun.current_step_size() << " (Train loss: " << previous_train_loss << " -> " << train_loss << ")");
          }
        } else {
          update_fun.update_step_size(decrease_step_factor); // bad epoch. reduce step size
          if (worker_id == 0) {
            ADLOG("Bold driver: Decrease step size to " << update_fun.current_step_size() << " (Train loss: " << previous_train_loss << " -> " << train_loss << ")");
          }
        }
      }

      previous_train_loss = train_loss;
    }

    // maximum time
    if (sw["runtime"].elapsed_s() > max_runtime ||
        sw["runtime"].elapsed_s() + sw["epoch"].elapsed_s() > max_runtime * 1.05) {
      ADLOG("Worker " << worker_id << " stops after epoch " << epoch << " because max. time is reached: " << sw["runtime"].elapsed_s() << "s (+1 epoch) > " << max_runtime << "s (epoch: " << sw["epoch"].elapsed_s() << "s)");
      break;
    }
  }

  // make sure all workers finished
  LLOG("Worker " << worker_id << " done.");
  kv.Finalize();
  Finalize(customer_id, false);
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("dataset,d", po::value<std::string>(&dataset), "Dataset to train from")
    ("num_rows", po::value<int>(&num_rows), "number of rows in the dataset")
    ("num_cols", po::value<int>(&num_cols), "number of columns in the dataset")
    ("rank,r", po::value<uint>(&mf_rank), "Rank of matrix factorization")
    ("epochs,e", po::value<int>(&epochs), "Number of epochs to run")
    ("lambda,l", po::value<double>(&lambda)->default_value(0.05), "Regularization parameter lambda")
    ("eps", po::value<double>(&initial_step_size)->default_value(0.001), "Initial step size")
    ("bold_driver", po::value<bool>(&bold_driver)->default_value(true), "Use bold driver for step size selection")
    ("increase_step_factor", po::value<double>(&increase_step_factor)->default_value(1.05), "Factor to increase step size after successful epoch")
    ("decrease_step_factor", po::value<double>(&decrease_step_factor)->default_value(0.5), "Factor to decrease step size after unsuccessful epoch")
    ("shared_memory", po::value<bool>(&shared_memory)->default_value(true), "access local parameters via shared memory")
    ("algorithm", po::value<Alg>(&algorithm)->default_value(Alg::dsgd), "which algorithm to use. options: (1) dsgd = DSGD, (2) columnwise = access data points by column, (3) plain_sgd")
    ("disable_localize_blocks", po::value<bool>(&disable_localize_blocks)->default_value(false), "disable localizes for H blocks (relevant only for DSGD)")
    ("disable_localize_left", po::value<bool>(&disable_localize_left)->default_value(false), "disable localization of left matrix (default: off, i.e., do localize left)")
    ("prelocalize_steps", po::value<size_t>(&prelocalize_steps)->default_value(0), "localize parameters N steps ahead of time")
    ("prelocalize_groupsize", po::value<size_t>(&prelocalize_groupsize)->default_value(1000), "number of data points to pre-localize in one call")
    ("replicate", po::value<size_t>(&replicate)->default_value(0), "number of keys to treat as hotspots (i.e., replicate)")
    ("num_threads,t", po::value<int>(&num_threads)->default_value(1), "number of worker threads to run")
    ("wor_blocks", po::value<bool>(&use_wor_block_schedule)->default_value(true), "use WOR schedule for blocks")
    ("wor_points", po::value<bool>(&use_wor_point_schedule)->default_value(true), "use WOR schedule for data points")
    ("sync_push", po::value<bool>(&sync_push)->default_value(false), "synchronous or asynchronous (default) pushes")
    ("compute_loss", po::value<bool>(&compute_loss)->default_value(true), "compute loss")
    ("init_parameters", po::value<int>(&init_parameters)->default_value(2), "how to initialize parameters. 0: no init, 1: read factors from files, 2: draw random factors")
    ("model_seed", po::value<unsigned>(&model_seed)->default_value(134827), "seed for model generation")
    ("enforce_random_keys", po::value<bool>(&enforce_random_keys)->default_value(false), "enforce that keys are assigned randomly")
    ("early_stop", po::value<unsigned long>(&early_stop)->default_value(0), "stop an epoch early after N data points (for debugging)")
    ("max_runtime", po::value<long>(&max_runtime)->default_value(std::numeric_limits<long>::max()), "set a maximum run tim, after which the job will be terminated (in seconds)")
    ("prevent_full_model_pull", po::value<bool>(&prevent_full_model_pull)->default_value(false), "prevent a full model pull (slower, but makes it possible to run large models in memory constrained settings)")
    ;

  // add system options
  ServerT::AddSystemOptions(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (!vm.count("dataset") || !vm.count("rank") || !vm.count("epochs")) {
    cout << "Either the dataset, the MF rank, or the number of epochs to run was not specified. Usage:\n\n";
    cout << desc << "\n";
    return 1;
  }

  if (algorithm == Alg::dsgd && prelocalize_steps > 0) {
    cout << "Setting `prelocalize_steps` > 0 is invalid when using DSGD\n";
    cout << desc << "\n";
    return 1;
  }
  if (algorithm != Alg::dsgd && disable_localize_blocks) {
    cout << "`disable_localize_blocks` = true has no effect when not using DSGD\n";
    cout << desc << "\n";
    return 1;
  }
  if (algorithm == Alg::dsgd && replicate > 0) {
    cout << "Both DSGD and replication activated, which does not make sense\n";
    cout << desc << "\n";
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;
  num_keys = num_cols + num_rows + 50000;
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads);
  Postoffice::Get()->set_shared_memory_access(shared_memory);

  std::string role = std::string(getenv("DMLC_ROLE"));
  std::cout << "mf. " << role << ": " << epochs << " epochs on " << dataset << "\n";
  std::cout << "localization: disable blocks " << disable_localize_blocks << ". disable left " << disable_localize_left << "\n";
  std::cout << "replicate " << replicate << "\n";

  // calculate H key offset
  num_workers = atoi(Environment::Get()->find("DMLC_NUM_SERVER")) * num_threads;
  first_col_key = round(ceil(1.0*num_rows/num_workers)) * num_workers;

  // enforce random parameter allocation
  if (enforce_random_keys) {
    key_assignment.resize(num_keys-1); // exclude the loss key
    iota(key_assignment.begin(), key_assignment.end(), 0);
    srand(2); // enforce same seed among different ranks
    random_shuffle(key_assignment.begin(), key_assignment.end());
  }

  // set up replication
  vector<Key> hotspot_keys = determine_hotspot_keys(replicate);

  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, mf_rank);
    auto server = new ServerT(server_customer_id, handle, &hotspot_keys);
    RegisterExitCallback([server](){ delete server; });

    // make sure all servers are set up
    server->Barrier();

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_threads; ++i) {
      workers.push_back(std::thread(RunWorker, i+1, server));
      std::string name = std::to_string(ps::MyRank())+"-worker-"+std::to_string(ps::MyRank()*num_threads + i);
      SET_THREAD_NAME((&workers[workers.size()-1]), name.c_str());
    }

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);
  }
}
