#include "utils.h"
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
  WorkerT kv(customer_id, *server); // app_id, customer_id
  int worker_id = ps::MyRank()*num_local_workers+customer_id; // a unique id for this worker thread
  int num_workers = ps::NumServers() * num_local_workers;

  std::string prfx = std::string("Locality API, worker ") + std::to_string(worker_id) + ": ";
  std::stringstream errors {};

  assert(Postoffice::Get()->num_servers() == 3);

  // wait for all workers to boot up
  kv.Barrier();

  std::vector<Val> vals (4);
  std::vector<Key> keys (2);

  // check initial locality
  if (worker_id == 0) { // rank 0
    kv.PullIfLocal(0, &vals) || errors << prfx << "FAILED. Initial locality of key 0 is wrong" << "\n";
    kv.PullIfLocal(3, &vals) || errors << prfx << "FAILED. Initial locality of key 3 is wrong" << "\n";
    !kv.PullIfLocal(4, &vals) || errors << prfx << "FAILED. Initial locality of key 4 is wrong" << "\n";
    !kv.PullIfLocal(5, &vals) || errors << prfx << "FAILED. Initial locality of key 5 is wrong" << "\n";
    kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. Initial locality of key 6 is wrong" << "\n";
  }
  if (worker_id == num_workers-1) { // rank 2
    !kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Initial locality of key 1 is wrong" << "\n";
    !kv.PullIfLocal(3, &vals) || errors << prfx << "FAILED. Initial locality of key 3 is wrong" << "\n";
    !kv.PullIfLocal(4, &vals) || errors << prfx << "FAILED. Initial locality of key 4 is wrong" << "\n";
    kv.PullIfLocal(5, &vals) || errors << prfx << "FAILED. Initial locality of key 5 is wrong" << "\n";
    !kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. Initial locality of key 6 is wrong" << "\n";
  }

  kv.Barrier();

  // move some parameters
  if (worker_id == 0) {
    keys = {3, 4};
    kv.Wait(kv.Intent(keys, 1));
  }
  if (worker_id == num_workers-1) {
    keys = {1, 3, 6};
    kv.Wait(kv.Intent(keys, 1));
  }

  kv.advanceClock();

  kv.WaitSync();
  kv.Barrier();

  // check modified locality
  if (worker_id == 0) {
    kv.PullIfLocal(0, &vals) || errors << prfx << "FAILED. Changed locality of key 0 is wrong" << "\n";
    kv.PullIfLocal(3, &vals) || errors << prfx << "FAILED. Changed locality of key 3 is wrong" << "\n";
    kv.PullIfLocal(4, &vals) || errors << prfx << "FAILED. Changed locality of key 4 is wrong" << "\n";
    !kv.PullIfLocal(5, &vals) || errors << prfx << "FAILED. Changed locality of key 5 is wrong" << "\n";
    !kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. Changed locality of key 6 is wrong" << "\n";
  }
  if (worker_id == num_workers-1) {
    !kv.PullIfLocal(0, &vals) || errors << prfx << "FAILED. Changed locality of key 0 is wrong" << "\n";
    kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. Changed locality of key 1 is wrong" << "\n";
    kv.PullIfLocal(3, &vals) || errors << prfx << "FAILED. Changed locality of key 3 is wrong" << "\n";
    !kv.PullIfLocal(4, &vals) || errors << prfx << "FAILED. Changed locality of key 4 is wrong" << "\n";
    kv.PullIfLocal(5, &vals) || errors << prfx << "FAILED. Changed locality of key 5 is wrong" << "\n";
    kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. Changed locality of key 6 is wrong" << "\n";
  }

  // test locality after replica destruction
  kv.advanceClock();
  kv.WaitSync();

  if (worker_id == 0) {
    kv.PullIfLocal(0, &vals) || errors << prfx << "FAILED. No-intent locality of key 0 is wrong" << "\n";
    kv.PullIfLocal(4, &vals) || errors << prfx << "FAILED. No-intent locality of key 4 is wrong" << "\n";
    !kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. No-intent locality of key 6 is wrong" << "\n";
  }
  if (worker_id == num_workers-1) {
    kv.PullIfLocal(1, &vals) || errors << prfx << "FAILED. No-intent locality of key 1 is wrong" << "\n";
    kv.PullIfLocal(5, &vals) || errors << prfx << "FAILED. No-intent locality of key 5 is wrong" << "\n";
    kv.PullIfLocal(6, &vals) || errors << prfx << "FAILED. No-intent locality of key 6 is wrong" << "\n";
  }

  // IsFinished tests
  if (worker_id == 0) {
    for(auto i = 0; i != 10; ++i) {
      keys[0] = 7;
      auto ts = kv.Pull(keys, &vals);
      ts != -1 || errors << prfx << "FAILED. Remote request returned -1" << "\n";

      kv.Wait(ts);
      kv.IsFinished(ts) || errors << prfx << "FAILED. Remote request isn't finished after wait" << "\n";


      keys[0] = 0; // 0 should be at this node throughout, so it should be local
      keys[1] = 4; // we have moved 4 to this node, so it should be local, too
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


  CHECK(!error) << "Test failed";

  return error;
}
