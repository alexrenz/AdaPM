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
#include "mf/loss.h"

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
Key  num_keys;
bool shared_memory;
bool use_localization;
bool localize_static;
int init_parameters;
bool enforce_random_keys;
std::vector<Key> key_assignment;

// Shared data
std::vector<Key> full_h_keys;
std::vector<ValT> full_h;

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
  int num_cols, num_rows;
  mf::DataPart data = read_training_data(dataset + string("train.mmc"), true, num_workers, worker_id, num_rows, num_cols);
  sw["read_data"].stop();
  LLOG("Finished reading train data in worker " << worker_id << " (" << sw["read_data"] << ")");
  int num_w_keys = data.num_rows_per_block() * num_workers;

  // allocate memory for factors in training loop
  std::vector<Key> factors_keys ( 2 );
  std::vector<ValT> factors ( factors_keys.size() * mf_rank );
  std::vector<ValT> factors_update ( factors.size() );

  // allocate memory for factors in loss computation (once per process)
  if (customer_id == 1) {
    full_h_keys.resize(num_cols);
    full_h.resize(num_cols * mf_rank);
    std::iota(full_h_keys.begin(), full_h_keys.end(), num_w_keys);
  }

  std::vector<Key> local_h_keys ( data.num_cols_per_block() );
  std::vector<Key> local_w_keys ( data.num_rows_per_block() );
  std::iota(local_w_keys.begin(), local_w_keys.end(), data.num_rows_per_block() * worker_id);
  std::vector<ValT> local_w ( local_w_keys.size() * mf_rank );

  boost::random::uniform_real_distribution<> factor_generator(0,1);

  // push factors into the parameter servers
  if (worker_id == 0 && init_parameters > 0) {
    // read and send H
    sw["read_factors"].start();
    if (init_parameters == 1) {
      // read factors from file
      full_h = read_dense_matrix<std::vector<ValT>>(dataset + "H.mma", false);
    } else if (init_parameters == 2) {
      // draw random factors
      for (size_t z=0; z!=full_h.size(); ++z) {
        full_h[z] = factor_generator(rng);
      }
    }
    kv.Wait(kv.Push(full_h_keys, full_h));
    sw["read_factors"].stop();
    LLOG("Finished reading and pushing H (" << sw["read_factors"] << ")");
  }

  if (init_parameters > 0 && worker_id == num_workers-1) {
    // read and send W
    sw["read_factors"].start();
    vector<Key> full_w_keys (num_rows);
    std::iota(full_w_keys.begin(), full_w_keys.end(), 0);
    std::vector<ValT> full_w;
    if (init_parameters == 1) {
      // read factors from files
      full_w = read_dense_matrix<std::vector<ValT>>(dataset + "W.mma", true);
    } else if (init_parameters == 2) {
      // draw random factors
      full_w.resize(full_w_keys.size() * mf_rank);
      for (size_t z=0; z!=full_w.size(); ++z) {
        full_w[z] = factor_generator(rng);
      }
    }
    // ADLOG("Push keys " << full_w_keys << " with vals " << full_w);
    kv.Wait(kv.Push(full_w_keys, full_w));
    sw["read_factors"].stop();
    LLOG("Finished reading and pushing W (" << sw["read_factors"] << ")");
  }
  kv.ResetStats();
  kv.Barrier(); // wait until all parameters are in the server

  // Localize local part of W
    // ADLOG("Worker " << worker_id << " localizes " << local_w_keys[0] << "--" << local_w_keys[local_w_keys.size()-1]);
  if (use_localization || localize_static) {
    kv.Wait(kv.Localize(local_w_keys));
  }

  // Compute initial loss
  std::vector<Key> loss_key { static_cast<Key>((data.num_cols_per_block() + data.num_rows_per_block()) * num_workers) };
  std::vector<ValT> loss_vec (mf_rank);
  ValT current_local_loss, previous_local_loss, previous_global_loss;
  if (compute_loss) {
    int t_pullw = kv.Pull(local_w_keys, &local_w);
    if (customer_id == 1) { // fetch full h once per process
      kv.Wait(kv.Pull(full_h_keys, &full_h));
    }
    kv.Wait(t_pullw);
    kv.Barrier();

    // calculate local loss
    current_local_loss = loss_Nzsl_L2(data, local_w, full_h, lambda, mf_rank, worker_id);
    previous_local_loss = current_local_loss;
    ADLOG("Initial local loss (worker " << worker_id << "): " << current_local_loss);
    ADLOG("Initial step (worker " << worker_id << "): " << update_fun.current_step_size());

    // calculate global loss at PS
    loss_vec[0] = current_local_loss;
    kv.Wait(kv.Push(loss_key, loss_vec));
    kv.Barrier();
    kv.Wait(kv.Pull(loss_key, &loss_vec));
    previous_global_loss = loss_vec[0];
    if (worker_id == 0) ADLOG("(Ep.0) Loss:\t" << loss_vec[0]);
  }

  // additional barrier if loss computation is disabled
  kv.Barrier();

  /* Training loop */
  sw["total_training"].start();
  for(int epoch = 1; epoch != epochs+1; ++epoch) {
    if (worker_id == 0) ADLOG("(Ep." << epoch << ") Starting epoch " << epoch);
    sw["epoch"].start();

    // create WOR block schedule for this epoch, if desired (WOR = random, without replacement)
    if (use_wor_block_schedule) mf::newWorSchedule(block_schedule, epoch, rng);

    for(uint subepoch = 0; subepoch != num_workers; ++subepoch) {
      if (worker_id == 0) ADLOG("Subepoch " << epoch << "." << subepoch);

      // get the block this workers works in this epoch
      sw["comm"].resume();
      int h_block = block_schedule(subepoch, worker_id);
      if (use_localization) {
        std::iota(local_h_keys.begin(), local_h_keys.end(), num_w_keys + h_block * data.num_cols_per_block());
        // ADLOG("Worker " << worker_id << " localizes " << local_h_keys[0] << "," << local_h_keys[local_h_keys.size()-1]);
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
        factors_keys[0] = enforce_random_keys ? key_assignment[dp.i] : dp.i;
        factors_keys[1] = enforce_random_keys ? key_assignment[dp.j+num_w_keys] : dp.j+num_w_keys;
        kv.Wait(kv.Pull(factors_keys, &factors));

        // run update
        update_fun(dp.i, dp.j, factors, factors_update, dp.x, data.row_nnz(dp.i), data.col_nnz(dp.j));

        // send factor updates
        kv.Wait(kv.Push(factors_keys, factors_update));
      }

      sw["comp"].stop();

      // wait for all workers to finish the subepoch
      sw["barrier"].resume();
      kv.Barrier();
      sw["barrier"].stop();

      // ADLOG("Subepoch " << epoch << "." << subepoch << " took " << sw["subepoch"]);
    }

    sw["epoch"].stop();
    LLOG("(Ep." << epoch << ") Worker " << worker_id << " finished epoch " << epoch <<" (" << sw["epoch"] << ": " << sw["comm"] << " comm, "  << sw["comp"] << " comp, " << sw["barrier"] << " barrier)");
     sw["comm"].reset(); sw["comp"].reset(); sw["perm"].reset(); sw["barrier"].reset();

    if (compute_loss) {
      sw["loss"].start();
      // pull factors for loss computation and calculate loss
      int t_pullw = kv.Pull(local_w_keys, &local_w);
      if (customer_id == 1) { // fetch full h once per process
        kv.Wait(kv.Pull(full_h_keys, &full_h));
      }
      kv.Wait(t_pullw);
      kv.Barrier();

      current_local_loss = loss_Nzsl_L2(data, local_w, full_h, lambda, mf_rank, worker_id);
      // LLOG("(Ep." << epoch << ") Loss at worker " << worker_id << ": " << current_local_loss);

      // push the local loss to the parameter server and pull global loss
      loss_vec[0] = current_local_loss - previous_local_loss;
      kv.Wait(kv.Push(loss_key, loss_vec));
      previous_local_loss = current_local_loss;
      kv.Barrier();
      kv.Wait(kv.Pull(loss_key, &loss_vec));

      // bold driver: adapt step size
      if (bold_driver) {
        if (loss_vec[0] < previous_global_loss) {
          update_fun.update_step_size(increase_step_factor); // good epoch. increase step size slightly
        } else {
          update_fun.update_step_size(decrease_step_factor); // bad epoch. reduce step size
        }
      }
      previous_global_loss = loss_vec[0];

      sw["loss"].stop();
      if (worker_id == 0) {
        ADLOG("(Ep." << epoch << ") Loss:\t" << loss_vec[0] << " (" << sw["loss"] << "). Step size: " << update_fun.current_step_size());
        // LL << "(Ep." << epoch << ") Step:\t" << update_fun.current_step_size();
        //trace(epoch, sw["epoch"].elapsed_s(), update_fun.current_step_size(), loss_vec[0]);
      }
    }
  }

  // make sure all workers finished
  LLOG("Worker " << worker_id << " done.");
  kv.Barrier();
  Finalize(customer_id, false); // shut down worker customer without barrier
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("dataset,d", po::value<std::string>(&dataset), "Dataset to train from")
    ("rank,r", po::value<uint>(&mf_rank), "Rank of matrix factorization")
    ("epochs,e", po::value<int>(&epochs), "Number of epochs to run")
    ("lambda,l", po::value<double>(&lambda)->default_value(0.05), "Regularization parameter lambda")
    ("eps", po::value<double>(&initial_step_size)->default_value(0.001), "Initial step size")
    ("bold_driver", po::value<bool>(&bold_driver)->default_value(false), "Use bold driver for step size selection")
    ("increase_step_factor", po::value<double>(&increase_step_factor)->default_value(1.05), "Factor to increase step size after successful epoch")
    ("decrease_step_factor", po::value<double>(&decrease_step_factor)->default_value(0.5), "Factor to decrease step size after unsuccessful epoch")
    ("shared_memory", po::value<bool>(&shared_memory)->default_value(true), "access local parameters via shared memory")
    ("localize", po::value<bool>(&use_localization)->default_value(false), "localize local parameters")
    ("localize_static", po::value<bool>(&localize_static)->default_value(false), "use (static) data locality")
    ("num_keys,k", po::value<Key>(&num_keys)->default_value(1000), "number of keys")
    ("num_threads,t", po::value<int>(&num_threads)->default_value(1), "number of worker threads to run")
    ("wor_blocks", po::value<bool>(&use_wor_block_schedule)->default_value(true), "use WOR schedule for blocks")
    ("wor_points", po::value<bool>(&use_wor_point_schedule)->default_value(true), "use WOR schedule for data points")
    ("compute_loss", po::value<bool>(&compute_loss)->default_value(true), "compute loss")
    ("init_parameters", po::value<int>(&init_parameters)->default_value(1), "how to initialize parameters. 0: no init, 1: read factors from files, 2: draw random factors")
    ("enforce_random_keys", po::value<bool>(&enforce_random_keys)->default_value(false), "enforce that keys are assigned randomly")
    ;

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

  // cannot localize or evaluate when randomizing keys
  if (enforce_random_keys && (compute_loss || use_localization)) {
    cout << "At the moment, when enforcing random keys, there is no support for loss computation and localization.\n\n";
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;
  Postoffice::Get()->enable_dynamic_allocation(num_keys+2, num_threads);
  Postoffice::Get()->set_shared_memory_access(shared_memory);

  std::string role = std::string(getenv("DMLC_ROLE"));
  std::cout << "mf. " << role << ": " << epochs << " epochs on " << dataset << "\n";
  std::cout << "localization: dynamic " << use_localization << ". static " << localize_static << "\n";

  // enforce random parameter allocation
  if (enforce_random_keys) {
    key_assignment.resize(num_keys);
    iota(key_assignment.begin(), key_assignment.end(), 0);
    srand(2); // enforce same seed among different ranks
    random_shuffle(key_assignment.begin(), key_assignment.end());
  }

  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, mf_rank);
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server](){ delete server; });

    num_workers = ps::NumServers() * num_threads;

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker, i+1, server));

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // stop the server
    Finalize(server_customer_id, true);
  }
}
