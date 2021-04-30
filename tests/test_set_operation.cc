#include "utils.h"
#include "ps/ps.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <array>
#include <assert.h>

using namespace ps;

// handle and server types
typedef long ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;


// Config
int num_keys = 36;
size_t vpk = 2;

bool error = false;

int num_local_workers = 2;


template <typename Val>
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id

  int worker_id = ps::MyRank()*num_local_workers+customer_id-1; // a unique id for this worker thread
  auto num_workers = Postoffice::Get()->num_servers() * num_local_workers;

  // wait for all workers to boot up
  kv.Barrier();

  util::Stopwatch duration {};
  duration.start();
  auto rank = ps::MyRank();
  unsigned long pullpush_runs = 100;
  unsigned long check_runs = 50;

  // Stress test
  // all workers push and pull values all the time
  std::vector<Key> keys1 {9};
  std::vector<Val> vals1 (keys1.size()*vpk);
  std::vector<Val> vals2 (keys1.size()*vpk);
  std::vector<Val> vals3 (keys1.size()*vpk);
  std::vector<Val> correctvals (keys1.size()*vpk);
  std::vector<Val> testvals (keys1.size()*vpk);
  std::vector<int> ts;
  std::vector<int> ts_pull;
  std::vector<int> ts_localize;

  for(unsigned long checkrun=0; checkrun!=check_runs; ++checkrun) {
    // Set a value
    if (worker_id == 0) {
      correctvals[0] = rand();
      correctvals[1] = rand();
      kv.Wait(kv.Set(keys1, correctvals));
    }

    // Check that set is correct
    if (worker_id == 0) {
      kv.Wait(kv.Pull(keys1, &testvals));
      if(correctvals != testvals) {
        ADLOG("Run " << checkrun << ", initial check failed: actual " << testvals << ", correct " << correctvals);
        error = true;
      }
    }
    kv.Barrier();

    // push a couple of values
    for(unsigned long run=0; run!=pullpush_runs; ++run) {
      if (run % 10000 == 0) {
        std::stringstream s;
        s << "w" << rank << ":" << customer_id << ": run " << run;
      }
      if(run % 1000 == 0) { // chaos: everyone tries to localize constantly
        ts_localize.push_back(kv.Localize(keys1));
      }

      vals1[0] = worker_id;
      vals1[1] = worker_id * 2;
      ts.push_back(kv.Push(keys1, vals1));
      ts_pull.push_back(kv.Pull(keys1, &vals2));
    }

    // Wait for all pushs and pulls
    for(auto timestamp : ts) kv.Wait(timestamp);
    for(auto timestamp : ts_pull) kv.Wait(timestamp);
    kv.Barrier();

    // Check that the current value is correct
    if (worker_id == 0) {
      kv.Wait(kv.Pull(keys1, &testvals));
      for (auto w=0; w!=num_workers; ++w) {
        correctvals[0] += w * pullpush_runs;
        correctvals[1] += w*2 * pullpush_runs;
      }
      if(correctvals != testvals) {
        ADLOG("Run " << checkrun << ", end check failed: actual " << testvals << ", correct " << correctvals);
        error = true;
      }
    }
  }

  // stop the program if we detected a mistake
  assert(!error);
  // otherwise the test is passed
  ADLOG("Test PASSED");

  // wind down
  kv.WaitAll();
  ADLOG("Worker " << rank << ":" << customer_id << " is finished ");
  kv.Barrier();
  Finalize(customer_id, barrier);
}

int main(int argc, char *argv[]) {
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_local_workers, false);

  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    auto server = new ServerT(num_keys, vpk);
    RegisterExitCallback([server](){ delete server; });

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
    }

    // stop the server
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }

  return error;
}
