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


// Config
bool coloc = false;
int num_keys = 36;
size_t vpk = 2;

bool error = false;

bool replicate; // set by command line arg


template <typename Val>
void RunWorker(int customer_id, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id

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
    // if(run == runs / NumServers() * rank && customer_id == 1) { // orderly consecutive localizes
    if(run % 1000 == 0) { // chaos: everyone tries to localize constantly
      ts_localize.push_back(kv.Localize(keys1));
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
  }

  // wait for all requests before we continue
  for(auto timestamp : ts) kv.Wait(timestamp);
  for(auto timestamp : ts_pull) kv.Wait(timestamp);
  for(auto timestamp : ts_localize) kv.Wait(timestamp);

  std::cout << "Worker " << rank << ":" << customer_id << " is finished " << std::endl;
  kv.WaitReplicaSync();
  kv.Barrier();
  duration.stop();

  if(rank == 0 && customer_id == 1) {
    kv.Wait(kv.Pull(keys1, &vals1));
    std::vector<uint> correct {static_cast<uint>(Postoffice::Get()->num_servers() * Postoffice::Get()->num_worker_threads() * runs), 0};
    correct[1] = correct[0] * 2;
    std::cout << "--------------------------\n";
    std::cout << "Result:   " << vals1 << "\n";
    std::cout << "Correct:  " << correct << "\n";
    std::cout << "Time:  " << duration << "\n";
    if (static_cast<uint>(vals1[0]) == correct[0] &&
        vals1[1] == 2*vals1[0]) {
      std::cout << "Dynamic Allocation" << (replicate ? " (with replication)" : "") << ": PASSED" << std::endl;
    } else {
      std::cout << "Dynamic Allocation" << (replicate ? " (with replication)" : "") << ": FAILED" << std::endl;
      error = true;
    }
    std::cout << "--------------------------" << std::endl;
  }

  kv.Finalize();
  Finalize(customer_id, false);
}

int main(int argc, char *argv[]) {
  int num_local_workers = 2;
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_local_workers, false);

  // Colocate servers and workers into one process?
  coloc = true;
  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // replication
    vector<Key> replicated_keys {};
    if(argc > 1 && strcmp("replicate", argv[1]) == 0) {
      ALOG("Replication: on");
      replicated_keys = {9, 12, 15};
      replicate = true;
    } else {
      ALOG("Replication: off");
      replicate = false;
    }

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, vpk);
    auto server = new ServerT(server_customer_id, handle, &replicated_keys);
    RegisterExitCallback([server](){ delete server; });

    // make sure all servers are set up
    server->Barrier();

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
      ADLOG("Customer r" << Postoffice::Get()->my_rank() << ":c" << w+1 << " joined");
    }

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }

  return error;
}
