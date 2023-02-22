#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <valarray>
#include <memory>
#include <set>
#include <unordered_set>
#include <cassert>

#pragma once

template<typename vT>
std::string str(vT& v) {
  std::stringstream ss;
  for(auto& e : v) {
    ss << "\t";
    ss << e;
  }
  return ss.str();
}

template<typename vT>
std::string str(vT& v, const size_t max) {
  std::stringstream ss;
  size_t num_elements = 0;
  for(auto& e : v) {
    ss << "\t";
    ss << e;
    ++num_elements;
    if (num_elements >= max) {
      ss << "\t...";
      break;
    }
  }
  return ss.str();
}

// print vector
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  std::stringstream ss;
  ss << str(v);
  os << ss.str();
  return os;
}

// print valarray
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::valarray<T>& v) {
  std::stringstream ss;
  ss << str(v);
  os << ss.str();
  return os;
}

// print set
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
  std::stringstream ss;
  for (auto elem : set) {
    ss << "\t" << elem;
  }
  os << ss.str();
  return os;
}

// print unordered set
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set) {
  std::stringstream ss;
  for (auto elem : set) {
    ss << "\t" << elem;
  }
  os << ss.str();
  return os;
}

namespace util {

  class Stopwatch {

    std::chrono::time_point<std::chrono::high_resolution_clock> begin;

    std::chrono::high_resolution_clock::duration _elapsed {};

  public:
    void start() {
      reset();
      begin = std::chrono::high_resolution_clock::now();
    }

    void resume() {
      begin = std::chrono::high_resolution_clock::now();
    }

    void stop() {
      auto finish = std::chrono::high_resolution_clock::now();
      _elapsed += (finish - begin);
    }

    void reset() {
      _elapsed = {};
    }

    long elapsed_ms() {
      return std::chrono::duration_cast<std::chrono::milliseconds>(_elapsed).count();
    }

    long elapsed_us() {
      return std::chrono::duration_cast<std::chrono::microseconds>(_elapsed).count();
    }

    long elapsed_ns() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(_elapsed).count();
    }

    double elapsed_s() {
      return 1.0*elapsed_ns()/1000000000;
    }
  };

  class Trace {
    std::string fname;

  public:
  Trace(std::string fn) :fname{fn} {}

    void clear() {
      // clear the trace file
      std::ofstream tracefile (fname, std::ofstream::trunc);
      tracefile.close();
    }

    void operator() (int epoch, double time_elapsed, double step_size, double loss, double accuracy=0) {
      std::ofstream tracefile (fname, std::ofstream::app);
      tracefile << epoch << "\t" << time_elapsed << "\t" << loss << "\t" << step_size << "\t" << accuracy << std::endl;
      tracefile.close();
    }
  };

}

std::ostream& operator<<(std::ostream& os, util::Stopwatch& sw) {
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(3);
  ss << sw.elapsed_s() << "s";
  os << ss.str();
  return os;
}


/**
 * \brief All-Reduce a vector of values among all workers via the parameter server.
          This is designed for occasional use (e.g., to sum local losses to a global loss), _not_ for performance.
          Each worker pushes its local value to a specific key, which then each worker pulls.
 */
template<typename Val, typename WorkerT, typename Key>
std::vector<Val> ps_allreduce(std::vector<Val> local_values, int worker_id, Key key, WorkerT& kv) {
  auto len = kv.GetLen(key);
  std::vector<Key> keys {key};
  std::vector<Val> global_values (len);

  // reset value in PS to zero
  if (worker_id == 0) {
    kv.Wait(kv.Pull(keys, &global_values));
    for (size_t i=0; i!=len; ++i) global_values[i] = -global_values[i];
    kv.Wait(kv.Push(keys, global_values));
  }
  kv.Barrier();

  // each worker pushes its component
  kv.Wait(kv.Push(keys, local_values));
  kv.Barrier();

  // get the loss
  kv.Wait(kv.Pull(keys, &global_values));
  return global_values;
}

/**
 * \brief All-Reduce a single scalar via the PS (a wrapper for the vector variant)
*/
template<typename Val, typename WorkerT, typename Key>
Val ps_allreduce(Val local, int worker_id, Key key, WorkerT& kv) {
  auto len = kv.GetLen(key);
  std::vector<Val> local_vector (len);
  local_vector[0] = local;

  auto global_values = ps_allreduce(local_vector, worker_id, key, kv);
  return global_values[0];
}

