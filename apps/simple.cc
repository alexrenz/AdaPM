#include "utils.h"
#include "ps/ps.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <thread>
#include <numeric>
#include <boost/program_options.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>
#include <unistd.h>
#include <bitset>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

using namespace ps;
using namespace std;

typedef double ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;

uint num_workers = -1;

// Config
size_t num_iterations = 0;
size_t num_threads = 0;
Key num_keys = 0;
size_t num_values_per_key = 0;

void RunWorker(int customer_id, ServerT* server=nullptr) {
  Start(customer_id);
  std::unordered_map<std::string, util::Stopwatch> sw {};
  WorkerT kv(0, customer_id, *server);

  int worker_id = ps::MyRank()*num_threads+customer_id-1; // a unique id for this worker thread

  std::vector<Key> keys (1);
  std::vector<ValT> values (num_values_per_key);
  std::iota(values.begin(), values.end(), 1);
  std::vector<ValT> values_pull (num_values_per_key);

  for (size_t x=0; x != num_iterations; ++x) {
    keys[0] = x;

    // push update
    kv.Wait(kv.Push(keys, values));

    // print current value
    kv.Wait(kv.Pull(keys, &values_pull));
    std::stringstream s;
    s << "Key " << x << " in worker " << worker_id << ": " << values_pull << "\n";
    std::cout << s.str();
  }

  kv.Finalize();
  Finalize(customer_id, false);
}


int process_program_options(const int argc, const char *const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("num_keys,k", po::value<Key>(&num_keys)->default_value(10), "number of parameters")
    ("num_threads,t", po::value<size_t>(&num_threads)->default_value(2), "number of worker threads to run (per process)")
    ("num_iterations,i", po::value<size_t>(&num_iterations)->default_value(4), "number of iterations to run")
    ("num_values_per_key,v", po::value<size_t>(&num_values_per_key)->default_value(3), "number of values per key")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  // check for required options
  // if (!vm.count("important_option2") || !vm.count("important_option2")) {
  //   cout << "Either important option 1 or important option 2 was not specified. Usage:\n\n";
  //   cout << desc << "\n";
  //   return 1;
  // }

  return 0;
}

int main(int argc, char *argv[]) {
  // Read cmd arguments
  int po_error = process_program_options(argc, argv);
  if(po_error) return 1;
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads);

  std::string role = std::string(getenv("DMLC_ROLE"));
  std::cout << "simple. Starting " << role << ": running " << num_iterations << " iterations on " << num_keys << " keys (" << num_values_per_key << " length) in " << num_threads << " threads\n";

  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (num_keys, num_values_per_key); // the handle specifies how the server handles incoming Push() and Pull() calls
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server](){ delete server; });

    num_workers = ps::NumServers() * num_threads;

    // run worker(s)
    std::vector<std::thread> workers {};
    for (size_t i=0; i!=num_threads; ++i)
      workers.push_back(std::thread(RunWorker, i+1, server));

    // wait for the workers to finish
    for (auto & w : workers)
      w.join();

    // stop the server
    Finalize(server_customer_id, true);
  }
}
