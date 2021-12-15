#include "../apps/utils.h"
#include "ps/ps.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <array>

/*
  Tests for locality-conscious part of the parameter server API:
  PullIfLocal()
  IsFinished()
 */

using namespace ps;

// handle and server types
typedef long ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;


// Config
int num_keys = 36;
size_t vpk = 1;
int num_local_workers = 2;

bool error = false;


template <typename Val>
void RunWorker(int customer_id, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_local_workers+customer_id-1; // a unique id for this worker thread
  int num_workers = ps::NumServers() * num_local_workers;

  std::string prfx = std::string("Locality API, worker ") + std::to_string(worker_id) + ": ";
  std::stringstream errors {};

  // wait for all workers to boot up
  kv.Barrier();

  vector<Val> vals (2);
  vector<Key> keys (1);

  // check initial locality
  if (worker_id == 0) {
    kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Initial locality of key 1 is wrong" << "\n";
    !kv.PullIfLocal(16, &vals) || errors << prfx << "FAILED. Initial locality of key 16 is wrong" << "\n";
    !kv.PullIfLocal(32, &vals) || errors << prfx << "FAILED. Initial locality of key 32 is wrong" << "\n";
  }
  if (worker_id == num_workers-1) {
    !kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Initial locality of key 1 is wrong" << "\n";
    !kv.PullIfLocal(16, &vals) || errors << prfx << "FAILED. Initial locality of key 16 is wrong" << "\n";
    kv.PullIfLocal(32, &vals) || errors << prfx << "FAILED. Initial locality of key 32 is wrong" << "\n";
  }

  kv.Barrier();

  // move some parameters
  if (worker_id == 0) {
    keys[0] = 16;
    kv.Wait(kv.Localize(keys));
  }
  if (worker_id == num_workers-1) {
    keys[0] = 1;
    kv.Wait(kv.Localize(keys));
  }

  kv.Barrier();

  // check modified locality
  if (worker_id == 0) {
    !kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Changed locality of key 1 is wrong" << "\n";
    kv.PullIfLocal(16, &vals) || errors << prfx << "FAILED. Changed locality of key 16 is wrong" << "\n";
    !kv.PullIfLocal(32, &vals) || errors << prfx << "FAILED. Changed locality of key 32 is wrong" << "\n";
  }
  if (worker_id == num_workers-1) {
    kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Changed locality of key 1 is wrong" << "\n";
    !kv.PullIfLocal(16, &vals) || errors << prfx << "FAILED. Changed locality of key 16 is wrong" << "\n";
    kv.PullIfLocal(32, &vals) || errors << prfx << "FAILED. Changed locality of key 32 is wrong" << "\n";
  }

  // IsFinished tests
  if (worker_id == 0) {
    for(auto i = 0; i != 100; ++i) {
      keys[0] = 15;
      auto ts = kv.Pull(keys, &vals);
      ts != -1 || errors << prfx << "FAILED. Remote request returned -1" << "\n";

      kv.Wait(ts);
      kv.IsFinished(ts) || errors << prfx << "FAILED. Remote isn't finished after wait" << "\n";


      keys[0] = 4;
      auto ts2 = kv.Pull(keys, &vals);
      kv.IsFinished(ts2) || errors << prfx << "FAILED. Local request isn't finished right away" << "\n";
      ts2 == -1 || errors << prfx << "FAILED. Local request returned a timestamp different from -1" << "\n";
    }
  }

  if(errors.str().empty()) {
    ALOG(prfx << "PASSED");
  } else {
    ALOG(errors.str());
  }


  kv.Finalize();
  Finalize(customer_id, false);
}

int main(int argc, char *argv[]) {
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_local_workers, false);

  // Colocate servers and workers into one process?
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
    auto server = new ServerT(server_customer_id, handle);
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


  CHECK(!error) << "Test failed";

  return error;
}
