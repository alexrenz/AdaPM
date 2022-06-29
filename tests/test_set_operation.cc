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
  WorkerT kv(customer_id, *server); // app_id, customer_id

  int worker_id = ps::MyRank()*num_local_workers+customer_id; // a unique id for this worker thread
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
        ALOG("Run " << checkrun << ", initial check failed: actual " << testvals << ", correct " << correctvals);
        error = true;
      }
      kv.advanceClock();
    }
    kv.Barrier();

    // push a couple of values
    for(unsigned long run=0; run!=pullpush_runs; ++run) {
      if (run % 10000 == 0) {
        std::stringstream s;
        s << "w" << rank << ":" << customer_id << ": run " << run;
      }
      if(run % 100 == 0) { // chaos: send intent some of the time for a couple of clocks
        ts_localize.push_back(kv.Intent(keys1, kv.currentClock()+10));
      }

      vals1[0] = worker_id;
      vals1[1] = worker_id * 2;
      ts.push_back(kv.Push(keys1, vals1));
      ts_pull.push_back(kv.Pull(keys1, &vals2));

      kv.advanceClock();
    }

    // make sure there is no intent anymore
    for (size_t i=0; i!=10; ++i) {
      kv.advanceClock();
    }

    // Wait for all pushs and pulls
    for(auto timestamp : ts) kv.Wait(timestamp);
    for(auto timestamp : ts_pull) kv.Wait(timestamp);
    kv.WaitSync();
    kv.Barrier();

    kv.WaitSync();
    kv.Barrier();


    // Check that the current value is correct
    if (worker_id == 0) {
      kv.Wait(kv.Pull(keys1, &testvals));
      for (auto w=0; w!=num_workers; ++w) {
        correctvals[0] += w * pullpush_runs;
        correctvals[1] += w*2 * pullpush_runs;
      }
      if(correctvals != testvals) {
        ALOG("Run " << checkrun << ", end check failed: actual " << testvals << ", correct " << correctvals);
        error = true;
      }
    }
  }

  if (error) {
    ALOG("Set operation, w " << worker_id << ": FAILED");
  } else {
    ALOG("Set operation, w " << worker_id << ": PASSED");
  }

  // wind down
  kv.WaitAll();
  kv.Barrier();
  kv.Finalize();
}

int main(int argc, char *argv[]) {
  Setup(num_keys, num_local_workers);

  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Scheduler();
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    auto server = new ServerT(vpk);
    RegisterExitCallback([server](){ delete server; });

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
    }

    // stop the server
    server->shutdown();

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }

  return error;
}
