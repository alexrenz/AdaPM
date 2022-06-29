#include "utils.h"
#include "ps/ps.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <array>

using namespace ps;

// handle and server types
typedef long ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;

// Workload settings
size_t num_threads = 4;
size_t num_keys = 1000;
size_t vpk = 5;
size_t num_samples_per_worker = 10000;
size_t num_keys_per_sample = 5;
size_t prep_sample_ahead = 50;

bool quick;

std::stringstream variant;

std::vector<std::vector<size_t>> frequencies;
std::vector<Key> sampling_sequence;
std::atomic<size_t> sampling_sequence_position {0};

inline Key SampleKey() {
  auto pos = sampling_sequence_position++;
  return sampling_sequence[pos % sampling_sequence.size()];
  // wrap around for pool (background thread might prepare more than the workers request)
  //   and local sampling (need more than the generated ones)
}


template <typename Val>
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  WorkerT kv(customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_threads+customer_id; // a unique id for this worker thread

  // localize keys away from their home node to make the test a bit more challenging
  if (customer_id == 0) {
    std::vector<Key> loc_keys {};
    for (Key k = ps::MyRank(); k < num_keys; k+=ps::NumServers()) {
      loc_keys.push_back(k);
      kv.Wait(kv.Intent(loc_keys, 0));
    }
  }

  if (worker_id == 0) {
    std::vector<Key> all_keys (num_keys);
    std::vector<Val> all_vals (all_keys.size() * vpk);
    for (unsigned int k=0; k!=num_keys; ++k) {
      all_keys[k] = k;
      for (unsigned int j=0; j!=vpk; ++j) {
        all_vals[k*vpk+j] = k*100+j;
      }
    }
    kv.Wait(kv.Push(all_keys, all_vals));
  }

  // wait for all workers to boot up
  kv.WaitSync();
  kv.Barrier();
  kv.WaitSync();

  util::Stopwatch duration {};
  duration.start();

  std::vector<Key> pull_keys (1);
  std::vector<Val> pull_vals (pull_keys.size() * vpk);
  std::vector<SampleID> sample_ids (num_samples_per_worker);

  long val_errors = 0;
  long val_checks = 0;
  long unique_errors = 0;
  long unique_checks = 0;

  std::mt19937 wgen {173727LU^worker_id};
  std::uniform_int_distribution<Key> num_keys_dist {1, vpk};

  // run the sampling workload
  size_t i_future = 0;
  for(size_t i=0; i!=num_samples_per_worker; ++i) {
    Clock futureClock = kv.currentClock()+i_future-i;
    // prep future samples
    while (i_future <= i+prep_sample_ahead && i_future < num_samples_per_worker) {
      sample_ids[i_future] = kv.PrepareSample(num_keys_per_sample * Sampling<ValT, WorkerT, ServerT>::reuse_factor_, futureClock);
      ++i_future;
    }

    std::vector<Key> all_pulled_keys {};

    // pull samples
    for (int keys_remaining = num_keys_per_sample * Sampling<ValT, WorkerT, ServerT>::reuse_factor_; keys_remaining>0; ) {

      // pull 1..keys_remaining keys from this sample
      int num_keys = num_keys_dist(wgen);
      if (num_keys > keys_remaining) {
        num_keys = keys_remaining;
      }
      pull_keys.resize(num_keys);
      pull_vals.resize(num_keys*vpk);

      kv.Wait(kv.PullSample(sample_ids[i], pull_keys, pull_vals));

      // val check
      for (int k=0; k!=num_keys; ++k) {
        frequencies[customer_id][pull_keys[k]] += 1;
        for (unsigned int z=0; z!=vpk; ++z) {
          if (pull_vals[k*vpk+z] != static_cast<ValT>(pull_keys[k]*100+z)) {
            ++val_errors;
            ALOG("ERROR in val check in worker " << worker_id << ": Pos " << z << " of key " << pull_keys[k] << " (key " << k << " in sample) should be " << pull_keys[k]*100+z << ", but is " << pull_vals[k*vpk+z]);
          }
          ++val_checks;
        }
      }

      all_pulled_keys.insert(all_pulled_keys.end(), pull_keys.begin(), pull_keys.end());

      keys_remaining -= num_keys;
    }

    // check that keys are unique (for without-replacement sampling)
    if (!Sampling<ValT, WorkerT, ServerT>::with_replacement_) {
      ++unique_checks;
      std::unordered_set<Key> pull_keys_set (all_pulled_keys.begin(), all_pulled_keys.end());
      if (all_pulled_keys.size() != pull_keys_set.size()) {
        ++unique_errors;
        ALOG("ERROR in unique check in worker " << worker_id << ": Pulled keys are not unique: " << all_pulled_keys);
      }
    }

    kv.advanceClock();
  }

  // val check output
  ALOG("Sampling (" << variant.str() << "), worker " << worker_id << " val check " << (val_errors != 0 ? "FAILED" : "PASSED") << " (" << 100.0 * val_errors / (val_checks + 0.00001) << "% of " << val_checks << " wrong)");

  // uniqueness check output
  if (!Sampling<ValT, WorkerT, ServerT>::with_replacement_) {
    ALOG("Sampling (" << variant.str() << "), worker " << worker_id << " unique check " << (unique_errors != 0 ? "FAILED" : "PASSED") << " (" << 100.0 * unique_errors / (unique_checks + 0.00001) << "% of " << unique_checks << " wrong)");
  }

  kv.Barrier();
  duration.stop();
  if (worker_id == 0) { ALOG("Sampling test took " << duration); }

  kv.Finalize();
}

int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ;

  // add system options
  ServerT::AddSystemOptions(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  return 0;
}

int main(int argc, char *argv[]) {
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;

  Setup(num_keys, num_threads);

  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Scheduler();
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    auto server = new ServerT(vpk);
    RegisterExitCallback([server](){ delete server; });

    // make sure all servers are set up
    server->Barrier();

    // sampling support
    server->enable_sampling_support(&SampleKey);
    size_t rank = ps::MyRank();
    auto num_servers = ps::NumServers();

    variant << Sampling<ValT, WorkerT, ServerT>::scheme << ", "
            << (Sampling<ValT, WorkerT, ServerT>::with_replacement_ ? "with" : "without") << " replacement, "
            << Sampling<ValT, WorkerT, ServerT>::reuse_factor_ << "x reuse, "
            << Sampling<ValT, WorkerT, ServerT>::pool_size_ << "ps"
      ;

    // set up sampling sequence
    auto num_total_samples = num_samples_per_worker * num_keys_per_sample * num_threads;
    sampling_sequence.reserve(num_total_samples);
    std::vector<size_t> correct_sampling_frequency (num_keys, 0);
    std::mt19937 gen {17727LU^rank};
    std::uniform_int_distribution<Key> dist  {0, num_keys-1};
    std::unordered_set<Key> no_sampling {};
    // have a couple of keys that we don't want to sample at all
    while (no_sampling.size() != num_keys / 3) {
      no_sampling.insert(dist(gen));
    }
    while (sampling_sequence.size() != num_total_samples) {
      auto k = dist(gen);
      if (no_sampling.find(k) != no_sampling.end()) { continue; } // reject "no sample" keys
      if (dist(gen) < k) { continue; } // create skew

      // use this sample
      sampling_sequence.push_back(k);
      correct_sampling_frequency[k] += 1;
    }

    // prep frequency count vectors
    frequencies.resize(num_threads);
    for(auto& f : frequencies) {
      f.resize(num_keys, 0);
    }

    // run worker(s)
    std::vector<std::thread> workers {};
    for (size_t i=0; i!=num_threads; ++i) {
      workers.push_back(std::thread(RunWorker<ValT>, i, false, server));
      std::string name = std::to_string(ps::MyRank())+"-worker-"+std::to_string(ps::MyRank()*num_threads + i);
      SET_THREAD_NAME((&workers[workers.size()-1]), name.c_str());
    }

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
    }

    // in local sampling, correct numbers are different
    if (Sampling<ValT, WorkerT, ServerT>::scheme == SamplingScheme::Local) {
      std::fill(correct_sampling_frequency.begin(), correct_sampling_frequency.end(), 0);
      size_t num_samples_total = 0;
      size_t pos = 0;
      while (num_samples_total != num_samples_per_worker * num_threads * num_keys_per_sample) {
        auto k = sampling_sequence[pos++ % sampling_sequence.size()];
        if (k % num_servers == rank) {
          correct_sampling_frequency[k] += 1;
          num_samples_total += 1;
        }
      }
    }

    // check sampling frequency (not for without replacement sampling)
    if (Sampling<ValT, WorkerT, ServerT>::with_replacement_) {
      // check
      size_t correct = 0;
      size_t errors = 0;
      for (size_t k=0; k!=num_keys; ++k) {
        size_t sampled = 0;
        for (size_t j=0; j!=num_threads; ++j) {
          sampled += frequencies[j][k];
        }
        if (sampled != correct_sampling_frequency[k] * Sampling<ValT, WorkerT, ServerT>::reuse_factor_) {
          ALOG("Rank " << rank  << ": key " << k << " was sampled " << sampled << " times, but should be " << correct_sampling_frequency[k]);
          ++errors;
        } else {
          ++correct;
        }
      }

      // output
      ALOG("Sampling (" << variant.str() << "), rank " << rank << ": frequency check " << (errors != 0 ? "FAILED" : "PASSED") << " (" << 100.0 * errors / (correct + errors + 0.00001) << "% of " << errors+correct << " wrong)");
    } else {
      ALOG("Sampling (" << variant.str() << "), rank " << rank << ": frequency not checked because without replacement sampling was enabled");
    }

    // stop the server
    server->shutdown();

  } else {
    LL << "Process started with unknown role '" << role << "'.";
  }

  return 0;
}
