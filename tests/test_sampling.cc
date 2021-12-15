#include "../apps/utils.h"
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
size_t vpk = 1;
size_t num_samples_per_worker = 10000;
size_t num_keys_per_sample = 5;
size_t prep_sample_ahead = 10;

bool quick;
bool replicate;


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
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_threads+customer_id-1; // a unique id for this worker thread

  // localize keys away from their home node to make the test a bit more challenging
  if (customer_id == 1) {
    std::vector<Key> loc_keys {};
    for (Key k = ps::MyRank(); k < num_keys; k+=ps::NumServers()) {
      loc_keys.push_back(k);
      kv.Wait(kv.Localize(loc_keys));
    }
  }

  // wait for all workers to boot up
  kv.Barrier();

  util::Stopwatch duration {};
  duration.start();

  std::vector<Key> pull_keys (1);
  std::vector<Val> pull_vals (pull_keys.size() * vpk);
  std::vector<SampleID> sample_ids (num_samples_per_worker);

  // run the sampling workload
  size_t i_future = 0;
  for(size_t i=0; i!=num_samples_per_worker; ++i) {
    // prep future samples
    while (i_future <= i+prep_sample_ahead && i_future < num_samples_per_worker) {
      sample_ids[i_future] = kv.PrepareSample(num_keys_per_sample * SamplingSupport<ValT, WorkerT>::reuse_factor_);
      ++i_future;
    }

    // pull samples
    for (size_t j = 0; j!=num_keys_per_sample * SamplingSupport<ValT, WorkerT>::reuse_factor_; ++j) {
      kv.Wait(kv.PullSample(sample_ids[i], pull_keys, pull_vals));
      frequencies[customer_id-1][pull_keys[0]] += 1;
    }
  }

  kv.Barrier();
  duration.stop();
  if (worker_id == 0) { ADLOG("Sampling test took " << duration); }

  kv.Finalize();
  Finalize(customer_id, barrier);
}

int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("replicate", po::bool_switch(&replicate)->default_value(false), "replicate")
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

  std::vector<Key> hotspot_keys {};
  if (replicate) {
    for(Key k=0; k<num_keys; k+=5) {
      hotspot_keys.push_back(k);
    }
    ALOG("Replication: on (" << hotspot_keys.size() << " of " << num_keys << " keys)");
  } else {
    ALOG("Replication: off");
  }

  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads, false);

  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, vpk);
    auto server = new ServerT(server_customer_id, handle, &hotspot_keys);
    RegisterExitCallback([server](){ delete server; });

    // make sure all servers are set up
    server->Barrier();

    // sampling support
    server->enable_sampling_support(&SampleKey);
    size_t rank = ps::MyRank();
    auto num_servers = ps::NumServers();

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
    for (size_t i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
    }

    // in local sampling, correct numbers are different
    if (SamplingSupport<ValT, WorkerT>::sampling_strategy == SamplingSupportType::OnlyLocal) {
      std::fill(correct_sampling_frequency.begin(), correct_sampling_frequency.end(), 0);
      size_t num_samples_total = 0;
      size_t pos = 0;
      while (num_samples_total != num_samples_per_worker * num_threads * num_keys_per_sample) {
        auto k = sampling_sequence[pos++ % sampling_sequence.size()];
        if (k % num_servers == rank || (replicate && std::find(hotspot_keys.begin(), hotspot_keys.end(), k) != hotspot_keys.end()) ) {
          correct_sampling_frequency[k] += 1;
          num_samples_total += 1;
        }
      }
    }

    // check
    size_t correct = 0;
    size_t errors = 0;
    for (size_t k=0; k!=num_keys; ++k) {
      size_t sampled = 0;
      for (size_t j=0; j!=num_threads; ++j) {
        sampled += frequencies[j][k];
      }
      if (sampled != correct_sampling_frequency[k] * SamplingSupport<ValT, WorkerT>::reuse_factor_) {
        ALOG("Rank " << rank  << ": key " << k << " was sampled " << sampled << " times, but should be " << correct_sampling_frequency[k]);
        ++errors;
      } else {
        ++correct;
      }
    }

    // output
    std::stringstream reptype;
    if (replicate) {
      reptype << "(with ";
      reptype <<  ReplicaManager<ValT,HandleT>::get_rsm();
      reptype << " replication)";
    }
    std::stringstream s;
    s << SamplingSupport<ValT, WorkerT>::sampling_strategy
      << (SamplingSupport<ValT, WorkerT>::postpone_nonlocal_ ? " postpone" : "") << ", "
      << SamplingSupport<ValT, WorkerT>::reuse_factor_ << "x reuse, "
      << SamplingSupport<ValT, WorkerT>::group_size_ << "gs"
      ;
    ALOG("Sampling (" << s.str() << ") " << reptype.str() << ", rank " << rank << ": " << (errors != 0 ? "FAILED" : "PASSED") << " (" << 100.0 * errors / (correct + errors) << "% of " << errors+correct << " wrong)");

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unknown role '" << role << "'.";
  }

  return 0;
}
