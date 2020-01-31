#include "../apps/utils.h"
#include "ps/ps.h"
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <array>
#include <boost/program_options.hpp>

/*
  Parameter Server Access Times Microbenchmark

  Runs `num_accesses` pull and push calls to `num_keys` keys, sampled uniformly and measures time necessary to do so
*/

using namespace ps;

// handle and server types
typedef char ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;


// Config
int num_keys = 0;
int num_accesses = 0;
int num_threads = 0;
int value_length;
bool access_remote;
bool sc;


template <typename Val>
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id

  // init
  std::vector<Key> key (1);
  std::vector<ValT> val_push (value_length, 1);
  std::vector<ValT> val_pull (value_length);
  std::vector<Key> access_sequence (num_accesses);

  int rank = MyRank();
  int worker_id = rank*num_threads+customer_id-1; // a unique id for this worker thread


  if (worker_id == 0) {

    srand(7);
    for (int i = 0; i < num_accesses; ++i) {
      access_sequence[i] = rand() % ((num_keys-2)/2);
      if (access_remote) {
        access_sequence[i] += num_keys/2;
      }
    }

    util::Stopwatch sw {};
    sw.start();
    for (int i = 0; i < num_accesses; i=i+2) {
      // pull access
      key[0] = access_sequence[i];
      // ADLOG("Pull key " << key[0]);
      kv.Wait(kv.Pull(key, &val_pull));

      // push access
      key[0] = access_sequence[i+1];
      // ADLOG("Push key " << key[0]);
      kv.Wait(kv.Push(key, val_push));
    }
    sw.stop();
    ADLOG("Run time: " << sw.elapsed_ns() << "ns");
    ADLOG("Time per operation: " << 1.0 * sw.elapsed_ns() / num_accesses << "ns");
  }

  kv.Barrier();
  Finalize(customer_id, barrier);
}

int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("num_keys,k", po::value<int>(&num_keys)->default_value(1000), "number of keys in parameter server")
    ("value_length", po::value<int>(&value_length)->default_value(4), "length of the value")
    ("num_accesses", po::value<int>(&num_accesses)->default_value(100000), "Number of parameter accesses")
    ("num_threads,t", po::value<int>(&num_threads)->default_value(1), "number of worker threads to run")
    ("sc", po::value<bool>(&sc)->default_value(true), "use short circuit for local updates")
    ("access_remote", po::value<bool>(&access_remote)->default_value(false), "access remote PS")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {

  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;
  Postoffice::Get()->enable_dynamic_allocation(num_keys+2, num_threads);
  Postoffice::Get()->set_shortcircuit(sc);

  std::string role = std::string(getenv("DMLC_ROLE"));
  std::cout << "mb_ps_calls " << role << ": " << num_accesses << " accesses to " << num_keys << " keys (" << value_length << " value length, " << sizeof(ValT) * value_length << " bytes)\n";
  std::cout << "access_remote: " << access_remote << "\n";

  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, value_length);
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server](){ delete server; });

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // debug
    ADLOG("[s" << ps::MyRank() << "]:   " << server->resp_local << "/" << server->resp_total << " local responses");

    // stop the server
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unkown role '" << role << "'.";
  }
}
