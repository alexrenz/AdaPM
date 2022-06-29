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


// Config
int num_keys = 36;
size_t vpk = 2;

bool error = false;



template <typename Val>
void RunWorker(int customer_id, ServerT* server=nullptr) {
  WorkerT kv(customer_id, *server); // app_id, customer_id

  // wait for all workers to boot up
  kv.Barrier();

  util::Stopwatch duration {};
  duration.start();
  auto rank = ps::MyRank();
  unsigned long runs = 100000;

  // Stress test
  // all workers push and pull values all the time
  std::vector<Key> keys1 {9};
  std::vector<Val> vals1 (keys1.size()*vpk);
  std::vector<Val> vals2 (keys1.size()*vpk);
  std::vector<Val> vals3 (keys1.size()*vpk);
  std::vector<int> ts;
  std::vector<int> ts_pull;
  std::vector<int> ts_localize;
  for(unsigned long run=0; run!=runs; ++run) {
    if (run % 10000 == 0) {
      std::stringstream s;
      s << "w" << rank << ":" << customer_id << ": run " << run;
      // std::cout << s.str() << std::endl;
    }
    // if(run == runs / NumServers() * rank && customer_id == 0) { // orderly consecutive localizes
    if(rand() % 50 == 0) { // from time to time, send intent
      kv.Intent(keys1, kv.currentClock()+10, kv.currentClock()+40);
      // TEMP: no waiting for intent
    }

    vals1[0] = 1; //rand();
    vals1[1] = 2; //rand();
    ts.push_back(kv.Push(keys1, vals1));
    ts_pull.push_back(kv.Pull(keys1, &vals2));

    // sync
    // kv.Wait(ts.back());
    // bounded async
    // if (run>10) {
    //   kv.Wait(ts[ts.size()-10]);
    //   kv.Wait(ts_pull[ts_pull.size()-10]);
    // }
    // full async
    // don't wait at all
    kv.advanceClock();
  }

  // wait for all requests before we continue
  for(auto timestamp : ts) kv.Wait(timestamp);
  for(auto timestamp : ts_pull) kv.Wait(timestamp);
  // for(auto timestamp : ts_localize) kv.Wait(timestamp);

  std::cout << "Worker " << rank << ":" << customer_id << " is finished " << std::endl;
  kv.WaitSync();
  kv.Barrier();
  kv.WaitSync();
  duration.stop();

  if(rank == 0 && customer_id == 0) {
    kv.Wait(kv.Pull(keys1, &vals1));
    std::vector<uint> correct {static_cast<uint>(Postoffice::Get()->num_servers() * Postoffice::Get()->num_worker_threads() * runs), 0};
    correct[1] = correct[0] * 2;
    std::cout << "--------------------------\n";
    std::cout << "Result:   " << vals1 << "\n";
    std::cout << "Correct:  " << correct << "\n";
    std::cout << "Time:  " << duration << "\n";
    if (static_cast<uint>(vals1[0]) == correct[0] &&
        vals1[1] == 2*vals1[0]) {
      std::cout << "Dynamic Allocation: PASSED" << std::endl;
    } else {
      std::cout << "Dynamic Allocation: FAILED" << std::endl;
      error = true;
    }
    std::cout << "--------------------------" << std::endl;
  }

  kv.WaitSync();

  kv.Finalize();
}

int main(int argc, char *argv[]) {
  int num_local_workers = 2;
  Setup(num_keys, num_local_workers);

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

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i, server));

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
