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
int num_threads = 4;
size_t num_keys = 10000;
size_t vpk = 10;
size_t runs = 5000;
size_t max_keys_at_a_time = 1000;


size_t max_concurrent_loops = 10;
long process_errors = 0;

bool quick;
bool replicate;


// generate a list of random keys (for now, a unique list) // TODO: include duplicates
std::vector<Key> random_keys(const size_t num_keys, const size_t max_n=2000) {
  size_t K = (rand() % (max_n-1)) + 1; // have at least 1 key
  // std::vector<Key> keys;
  // keys.reserve(K);
  std::unordered_set<Key> keys;
  for(size_t k=0; keys.size()!=K; ++k) {
    Key key = rand() % num_keys;
    // keys.push_back(key);
    keys.insert(key);
  }
  // for (int i : vec)
  //   s.insert(i);
  // vec.assign( s.begin(), s.end() );
  std::vector<Key> keys_vec(keys.begin(), keys.end());
  std::random_shuffle(keys_vec.begin(), keys_vec.end());
  return keys_vec;
}

template <typename Val>
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_threads+customer_id-1; // a unique id for this worker thread
  long worker_errors = 0;
  long worker_correct = 0;

  std::stringstream reptype;
  if (replicate) {
    reptype << " (with ";
    reptype <<  ReplicaManager<ValT,HandleT>::get_rsm();
    reptype << " replication)";
  }

  // wait for all workers to boot up
  kv.Barrier();

  util::Stopwatch duration {};
  duration.start();

  std::vector<Key> all_keys (num_keys);
  std::vector<Val> init_vals (num_keys*vpk);

  // initialize keys randomly
  std::mt19937 gen(7);
  std::uniform_int_distribution<ValT> valdist(-5000000,5000000);
  for(size_t k=0; k!=num_keys; ++k) {
    all_keys[k] = k;
    for(size_t z=0; z!=vpk; ++z) {
      auto r = valdist(gen);
      init_vals[k*vpk+z] = r;
    }
  }
  if (worker_id == 0) {
    kv.Wait(kv.Push(all_keys, init_vals));
  }
  kv.WaitReplicaSync();
  kv.Barrier();

  std::vector<std::vector<Key>> push_keys (runs);
  std::vector<std::vector<Val>> push_vals (runs);
  std::vector<std::vector<Key>> pull_keys (runs);
  std::vector<std::vector<Val>> pull_vals (runs);
  std::vector<std::vector<Key>> loc_keys (runs);
  std::vector<int> pull_ts (runs);
  std::vector<int> push_ts (runs);


  // ---------------------------------------------------------------
  // pulls and localizes

  // run workload (pulls and localizes)
  for(size_t i=0; i!=runs; ++i) {
    // localize a random list of keys
    loc_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    kv.Localize(loc_keys[i]);

    // pull a random list of keys
    pull_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    pull_vals[i].resize(pull_keys[i].size() * vpk, 12);
    pull_ts[i] = kv.Pull(pull_keys[i], &(pull_vals[i]));

    // limit the number of concurrent operations
    if (i > max_concurrent_loops) {
      kv.Wait(pull_ts[i-max_concurrent_loops]);
    }
  }

  // check
  for (size_t run=0; run<runs; ++run) {
    kv.Wait(pull_ts[run]);
    for (size_t k=0; k!=pull_keys[run].size(); ++k) {
      Key key = pull_keys[run][k];
      for (size_t z=0; z!=vpk; ++z) {
        auto pulled = pull_vals.at(run).at(k*vpk+z);
        auto correct = init_vals[key*vpk+z];
        if (pulled != correct) {
          ADLOG(worker_id << ", run " << run << " Key " << key << ", pos " << z << ": pulled " << pulled << ", but has to be " << correct);
          ++worker_errors;
        } else {
          ++worker_correct;
        }
      }
    }
  }
  ADLOG("Worker " << worker_id << ": " << worker_errors << " of " << worker_errors+worker_correct << " vals wrong (" << 100.0 * worker_errors / (worker_correct + worker_errors) << "%)");

  process_errors += worker_errors;
  kv.WaitAll();
  kv.Barrier();
  ADLOG("Pulls and Localizes" << reptype.str() << ", w" << worker_id << ": " << (worker_errors != 0 ? "FAILED" : "PASSED") << " (" << worker_errors << " errors)");
  kv.Barrier();
  duration.stop();
  if (worker_id == 0) { ADLOG("Pulls and localizes test took " << duration); }




  // ---------------------------------------------------------------
  // monotonic pushs
  std::uniform_int_distribution<ValT> pushdist(1,1000);
  duration.start();

  // reset
  worker_errors = 0;
  worker_correct = 0;


  // run workload (pushs, pulls, localizes)
  for(size_t i=0; i!=runs; ++i) {
    // localize a random list of keys
    loc_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    kv.Localize(loc_keys[i]);

    // pull a random list of keys
    pull_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    pull_vals[i].resize(pull_keys[i].size() * vpk, 12);
    pull_ts[i] = kv.Pull(pull_keys[i], &(pull_vals[i]));

    // push a random list of keys
    push_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    push_vals[i].resize(push_keys[i].size() * vpk, 12);;
    for(size_t k=0; k!=push_keys[i].size(); ++k) {
      for(size_t z=0; z!=vpk; ++z) {
        auto r = pushdist(gen);
        push_vals[i][k*vpk+z] = r;
      }
    }
    push_ts[i] = kv.Push(push_keys[i], push_vals[i]);
    kv.Wait(push_ts[i]);

    // limit the number of concurrent operations
    if (i > max_concurrent_loops) {
      kv.Wait(pull_ts[i-max_concurrent_loops]);
    }
  }

  // check
  kv.WaitReplicaSync();
  kv.Barrier();

  for (size_t run=0; run<runs; ++run) {
    kv.Wait(pull_ts[run]);
    // check pulls
    for (size_t k=0; k!=pull_keys[run].size(); ++k) {
      Key key = pull_keys[run][k];
      for (size_t z=0; z!=vpk; ++z) {
        auto pulled = pull_vals.at(run).at(k*vpk+z);
        auto correct = init_vals[key*vpk+z];
        if (pulled < correct) {
          ADLOG(worker_id << ", run " << run << " Key " << key << ", pos " << z << ": pulled " << pulled << ", but has to be at least " << correct);
          ++worker_errors;
        } else {
          ++worker_correct;
        }
      }
    }
    // incorporate pushs
    for (size_t k=0; k!=push_keys[run].size(); ++k) {
      Key key = push_keys[run][k];
      for (size_t z=0; z!=vpk; ++z) {
        init_vals[key*vpk+z] += push_vals[run][k*vpk+z];
      }
    }
  }

  ADLOG("Worker " << worker_id << ": " << worker_errors << " of " << worker_errors+worker_correct << " vals wrong (" << 100.0 * worker_errors / (worker_correct + worker_errors) << "%)");

  process_errors += worker_errors;
  kv.WaitAll();
  kv.Barrier();
  ADLOG("Monotonic pushs" << reptype.str() << ", w" << worker_id << ": " << (worker_errors != 0 ? "FAILED" : "PASSED") << " (" << worker_errors << " errors)");
  kv.Barrier();
  duration.stop();
  if (worker_id == 0) { ADLOG("Monotonic pushs test took " << duration); }




  // ---------------------------------------------------------------
  // eventual consistency
  duration.start();

  // reset
  worker_errors = 0;
  worker_correct = 0;

  // current vals
  std::vector<Val> vals_before (num_keys*vpk);
  std::vector<Val> total_pushed (num_keys*vpk, 0);

  // pull
  kv.Wait(kv.Pull(all_keys, &vals_before));
  kv.Barrier();


  // run workload (pushs, localizes)
  for(size_t i=0; i!=runs; ++i) {
    // localize a random list of keys
    loc_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    kv.Localize(loc_keys[i]);

    // push a random list of keys
    push_keys[i] = random_keys(num_keys, max_keys_at_a_time);
    push_vals[i].resize(push_keys[i].size() * vpk, 12);;
    for(size_t k=0; k!=push_keys[i].size(); ++k) {
      Key key = push_keys[i][k];
      for(size_t z=0; z!=vpk; ++z) {
        auto r = pushdist(gen);
        push_vals[i][k*vpk+z] = r;
        total_pushed[key*vpk+z] -= r;
      }
    }
    push_ts[i] = kv.Push(push_keys[i], push_vals[i]);
    // kv.Wait(push_ts[i]);

    // limit the number of concurrent operations
    if (i > max_concurrent_loops) {
      kv.Wait(push_ts[i-max_concurrent_loops]);
    }
  }

  // revert the pushs of this worker
  kv.Push(all_keys, total_pushed);

  // check
  kv.WaitAll();
  kv.WaitReplicaSync();
  kv.Barrier();

  std::vector<Val> vals_after (num_keys*vpk);
  kv.Wait(kv.Pull(all_keys, &vals_after));

  for (size_t k=0; k!=all_keys.size(); ++k) {
    Key key = all_keys[k];
    for (size_t z=0; z!=vpk; ++z) {
      auto before = vals_before[key*vpk+z];
      auto after = vals_after[key*vpk+z];
      if (before != after) {
        ADLOG(worker_id << ", Key " << key << ", pos " << z << ", before!=after: " << before << "!=" << after << " (diff: " << after-before << ")");
        ++worker_errors;
      } else {
        ++worker_correct;
      }
    }
  }

  ADLOG("Worker " << worker_id << ": " << worker_errors << " of " << worker_errors+worker_correct << " vals wrong (" << 100.0 * worker_errors / (worker_correct + worker_errors) << "%)");

  process_errors += worker_errors;
  kv.WaitAll();
  kv.Barrier();
  ADLOG("Eventual consistency" << reptype.str() << ", w" << worker_id << ": " << (worker_errors != 0 ? "FAILED" : "PASSED") << " (" << worker_errors << " errors) ");
  kv.Barrier();
  duration.stop();
  if (worker_id == 0) { ADLOG("Eventual consistency test took " << duration); }



  kv.Finalize();
  Finalize(customer_id, barrier);
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("quick", po::bool_switch(&quick)->default_value(false), "quick mode")
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


  if(quick) {
    ALOG("Quick mode: run a lighter workload");
    runs = 500;
    num_threads = 2;
    num_keys = 100;
    max_keys_at_a_time = 20;
  } else {
    ALOG("Heavy mode: run full workload");
  }

  vector<Key> hotspot_keys {};
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

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
    }

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }

  return process_errors;
}
