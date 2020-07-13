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
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_local_workers+customer_id-1; // a unique id for this worker thread
  int num_workers = ps::NumServers() * num_local_workers;

  // wait for all workers to boot up
  kv.Barrier();

  vector<Val> vals (2);
  vector<Key> keys (1);

  // check initial locality
  if (worker_id == 0) {
    CHECK(kv.PullIfLocal(1, &vals));
    CHECK(!kv.PullIfLocal(16, &vals));
    CHECK(!kv.PullIfLocal(32, &vals));
  }
  if (worker_id == num_workers-1) {
    CHECK(!kv.PullIfLocal(1, &vals));
    CHECK(!kv.PullIfLocal(16, &vals));
    CHECK(kv.PullIfLocal(32, &vals));
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
    CHECK(!kv.PullIfLocal(1, &vals));
    CHECK(kv.PullIfLocal(16, &vals));
    CHECK(!kv.PullIfLocal(32, &vals));
  }
  if (worker_id == num_workers-1) {
    CHECK(kv.PullIfLocal(1, &vals));
    CHECK(!kv.PullIfLocal(16, &vals));
    CHECK(kv.PullIfLocal(32, &vals));
  }


  // IsFinished tests
  if (worker_id == 0) {
    for(auto i = 0; i != 100; ++i) {
      keys[0] = 15;
      auto ts = kv.Pull(keys, &vals);
      CHECK(!kv.IsFinished(ts)) <<  "remote request is finished although it should't be";
      kv.Wait(ts);
      CHECK(kv.IsFinished(ts)) << "remote request isn't finished although it should be";


      keys[0] = 4;
      auto ts2 = kv.Pull(keys, &vals);
      CHECK(kv.IsFinished(ts2)) <<  "local request isn't finished although it should be";
    }
  }


  kv.Barrier();
  Finalize(customer_id, barrier);
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

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
      ADLOG("Customer r" << Postoffice::Get()->my_rank() << ":c" << w+1 << " joined");
    }

    // stop the server
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }


  CHECK(!error) << "Test failed";

  return error;
}
