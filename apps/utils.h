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

#pragma once

using namespace std;

template<typename vT>
std::string str(vT v) {
  std::stringstream ss;
  for(size_t i = 0; i < v.size(); ++i) {
    ss << "\t";
    ss << v[i];
  }
  return ss.str();
}

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> v) {
  std::stringstream ss;
  ss << str(v);
  os << ss.str();
  return os;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, std::valarray<T> v) {
  std::stringstream ss;
  ss << str(v);
  os << ss.str();
  return os;
}

namespace util {

  class Stopwatch {

    chrono::time_point<std::chrono::high_resolution_clock> begin;

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
      return 1.0*elapsed_ms()/1000;
    }
  };

  class Trace {
    std::string fname;

  public:
  Trace(std::string fn) :fname{fn} {}

    void clear() {
      // clear the trace file
      ofstream tracefile (fname, ofstream::trunc);
      tracefile.close();
    }

    void operator() (int epoch, double time_elapsed, double step_size, double loss, double accuracy=0) {
      ofstream tracefile (fname, ofstream::app);
      tracefile << epoch << "\t" << time_elapsed << "\t" << loss << "\t" << step_size << "\t" << accuracy << endl;
      tracefile.close();
    }
  };

}

ostream& operator<<(ostream& os, util::Stopwatch& sw) {
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(3);
  ss << sw.elapsed_s() << "s";
  os << ss.str();
  return os;
}


// C++ 11 make_unique
template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
