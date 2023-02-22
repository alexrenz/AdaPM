#include <torch/extension.h>
#include <ps/ps.h>
#include <iostream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

typedef float ValT;
typedef ps::DefaultColoServerHandle<ValT> HandleT;
typedef ps::ColoKVServer<ValT, HandleT> ServerT;
typedef ps::ColoKVWorker<ValT, HandleT> WorkerT;

using namespace std;

namespace py = pybind11;


void setup(int num_keys, int num_threads, const std::string& use_techniques="", const int num_channels = -1) {
  // set specifically which techniques AdaPM should use to manage parameters
  if (use_techniques != "") {
    ps::Postoffice::Get()->set_management_techniques(
                           ps::tokenToMgmtTechniques(use_techniques));
  }

  // set specifically how many communication channels AdaPM should use
  if (num_channels != -1) {
    ps::Postoffice::Get()->set_num_channels(num_channels);
  }

  ps::Postoffice::Get()->setup(num_keys, num_threads);
}

void scheduler(int num_keys, int num_threads) {
  setup(num_keys, num_threads);
  ps::Scheduler();
}

void assert_correct_value_length(WorkerT& worker, long int* provided_keys, int provided_keys_length,
                                 int provided_value_length) {
  int needed_value_length = 0;
  for(int i=0; i!=provided_keys_length; ++i) {
    needed_value_length += worker.GetLen(provided_keys[i]);
  }
  if (needed_value_length != provided_value_length) {
    throw length_error("The provided value array does not match the size specified in the parameter server: " +
                       to_string(provided_value_length) +
                       " != " +
                       to_string(needed_value_length));
  }
}

void assert_keys_in_range(long int max_key, long int* provided_keys, int provided_keys_length) {
  for (int i = 0; i < provided_keys_length; i++) {
    if (provided_keys[i] >= max_key) {
      throw length_error("At least one of the provided keys (" +
                         to_string(provided_keys[i]) + 
                         ") is outside the key range [0, " +
                         to_string(max_key) + ")");
    }
  }
}


// sampling
std::mt19937 generator;

std::uniform_int_distribution<ps::Key> uniform;
inline ps::Key UniformSampling() {
  return uniform(generator);
}

std::uniform_real_distribution<long double> uniform_real;
ps::Key min_key, max_key;
inline ps::Key LogUniformSampling() {
  return static_cast<ps::Key>(exp(uniform_real(generator)*log(max_key-min_key+1))+min_key-1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup", &setup, "set up AdaPM",
        py::arg("num_keys"), py::arg("num_threads"),
        py::arg("use_techniques") = "", py::arg("num_channels") = -1);

  m.def("scheduler", &scheduler, "run scheduler");

  // Server

  py::class_<ServerT>(m, "Server")
    .def(py::init<int>())
    .def(py::init([](torch::Tensor& value_lengths) {
      auto t = value_lengths.to(torch::kInt64).contiguous(); // torch kInt64 should be a c++ long
      std::vector<size_t> lengths(t.data_ptr<long>(), t.data_ptr<long>() + t.numel());
      return new ServerT(lengths);
    }), py::arg("value_lengths").noconvert())

    // set up sampling
    .def("enable_sampling_support", [](ServerT& server,
                                       const std::string scheme,
                                       const bool with_replacement,
                                       const std::string distribution,
                                       const ps::Key min, // incl.
                                       const ps::Key max // excl.
                                       ){
      // This enables to use the AdaPM sampling support from the PyTorch bindings.
      // Out of the box, only some pre-provided sampling distributions are supported:
      //  - uniform: uniform sampling from a continuous range [min, max)  (incl. min, excl. max)
      //  - log-uniform: log-uniform sampling from a continuous range [min, max)  (incl. min, excl. max)

      generator = std::mt19937(ps::Postoffice::Get()->my_rank());
      ps::Sampling<ValT, WorkerT, ServerT>::with_replacement_ = with_replacement;

      // sampling scheme
      if      (scheme == "naive")  ps::Sampling<ValT, WorkerT, ServerT>::scheme = ps::SamplingScheme::Naive;
      else if (scheme == "preloc") ps::Sampling<ValT, WorkerT, ServerT>::scheme = ps::SamplingScheme::Preloc;
      else if (scheme == "pool")   ps::Sampling<ValT, WorkerT, ServerT>::scheme = ps::SamplingScheme::Pool;
      else if (scheme == "local")  ps::Sampling<ValT, WorkerT, ServerT>::scheme = ps::SamplingScheme::Local;
      else {
        ALOG("Unknown sampling scheme '" << scheme << "'. Aborting.");
        abort();
      }

      // sampling distribution
      if (distribution == "uniform") {
        uniform = std::uniform_int_distribution<ps::Key>{min, static_cast<int>(max-1)};
        server.enable_sampling_support(&UniformSampling, min, max);
      } else if (distribution == "log-uniform") {
        min_key = min;
        max_key = max;
        uniform_real = std::uniform_real_distribution<long double>{0, 1};
        server.enable_sampling_support(&LogUniformSampling, min, max);
      } else {
        ALOG("Unknown sampling distribution '" << distribution << "'. Aborting.");
        abort();
      }
    }, py::arg("scheme"), py::arg("with_replacement"), py::arg("distribution"), py::arg("min"), py::arg("max"))

    .def("barrier", [](ServerT& server){
      server.Barrier();
    })

    .def("shutdown", [](ServerT& server){
      server.shutdown();
    })

    .def("my_rank", [](ServerT& server){ return ps::MyRank(); })
    ;


  // Worker

  py::class_<WorkerT>(m, "Worker")
    .def(py::init([](const int customer_id, ServerT& server) {
      return new WorkerT(customer_id, server);
    }))

    .def("wait", [](WorkerT& worker, int timestamp){ worker.Wait(timestamp); })

    .def("waitall", [](WorkerT& worker){ worker.WaitAll(); })

    .def("finalize", [](WorkerT& worker){
      py::gil_scoped_release release;
      worker.Finalize();
    })


    // pull

    .def("pull", [](WorkerT& worker, torch::Tensor& keys, torch::Tensor& vals, bool async){
      ValT* val_ptr = static_cast<ValT*> (vals.data_ptr());
      size_t num_keys = keys.size(0);
      size_t num_vals = 1;
      for (int i = 0; i < vals.dim(); i++){
        num_vals *= vals.size(i);
      }
      long int* key_ptr = static_cast<long int*> (keys.data_ptr());
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Pull(key_ptr, num_keys,  val_ptr);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)

    .def("pull", [](WorkerT& worker, py::array_t<long int>& keys, py::array_t<ValT>& vals, bool async){
      auto val_buffer = vals.request();
      ValT* val_ptr = static_cast<ValT*> (val_buffer.ptr);
      auto key_buffer = keys.request();
      size_t num_keys = key_buffer.shape[0];
      size_t num_vals = 1;
      for (int i = 0; i < val_buffer.ndim; i++){
        num_vals *= val_buffer.shape[i];
      }
      long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Pull(key_ptr, num_keys,  val_ptr);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)


    // push

    .def("push", [](WorkerT& worker, torch::Tensor& keys, torch::Tensor& vals, bool async){
      ValT* val_ptr = static_cast<ValT*> (vals.data_ptr());
      size_t num_keys = keys.size(0);
      size_t num_vals = 1;
      for (int i = 0; i < vals.dim(); i++){
        num_vals *= vals.size(i);
      }
      long int* key_ptr = static_cast<long int*> (keys.data_ptr());
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Push(key_ptr, num_keys, val_ptr, false);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)

    .def("push", [](WorkerT& worker, py::array_t<long int>& keys, py::array_t<ValT>& vals, bool async){
      auto val_buffer = vals.request();
      ValT* val_ptr = static_cast<ValT*> (val_buffer.ptr);
      auto key_buffer = keys.request();
      size_t num_keys = key_buffer.shape[0];
      size_t num_vals = 1;
      for (int i = 0; i < val_buffer.ndim; i++){
        num_vals *= val_buffer.shape[i];
      }
      long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Push(key_ptr, num_keys, val_ptr, false);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)


    // set (a variant of push, use carefully with replication)

    .def("set", [](WorkerT& worker, torch::Tensor& keys, torch::Tensor& vals, bool async){
      ValT* val_ptr = static_cast<ValT*> (vals.data_ptr());
      size_t num_keys = keys.size(0);
      size_t num_vals = 1;
      for (int i = 0; i < vals.dim(); i++){
        num_vals *= vals.size(i);
      }
      long int* key_ptr = static_cast<long int*> (keys.data_ptr());
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Push(key_ptr, num_keys, val_ptr, true);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)

    .def("set", [](WorkerT& worker, py::array_t<long int>& keys, py::array_t<ValT>& vals, bool async){
      auto val_buffer = vals.request();
      ValT* val_ptr = static_cast<ValT*> (val_buffer.ptr);
      auto key_buffer = keys.request();
      size_t num_keys = key_buffer.shape[0];
      size_t num_vals = 1;
      for (int i = 0; i < val_buffer.ndim; i++){
        num_vals *= val_buffer.shape[i];
      }
      long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
      assert_correct_value_length(worker, key_ptr, num_keys, num_vals);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      auto ts = worker.Push(key_ptr, num_keys, val_ptr, true);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)


    // intent

    .def("intent", [](WorkerT& worker, torch::Tensor& keys, ps::Clock start, ps::Clock end){
      size_t num_keys = keys.size(0);
      long int* key_ptr = static_cast<long int*> (keys.data_ptr());
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      return worker.Intent(key_ptr, num_keys, start, end);
    }, py::arg("keys").noconvert(), py::arg("start"), py::arg("end")=0)

    .def("intent", [](WorkerT& worker, py::array_t<long int>& keys, ps::Clock start, ps::Clock end){
      auto key_buffer = keys.request();
      size_t num_keys = key_buffer.shape[0];
      long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
      assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
      return worker.Intent(key_ptr, num_keys, start, end);
    }, py::arg("keys").noconvert(), py::arg("start"), py::arg("end")=0)

    .def("advance_clock", [](WorkerT& worker){
      worker.advanceClock();
    })

    .def("current_clock", [](WorkerT& worker){
      return worker.currentClock();
    })


    // sampling

    .def("prepare_sample", [](WorkerT& worker, size_t K, ps::Clock start, ps::Clock end){
      return worker.PrepareSample(K, start, end);
    }, py::arg("K"), py::arg("start"), py::arg("end")=0)

    .def("pull_sample", [](WorkerT& worker, ps::SampleID id, torch::Tensor& keys, torch::Tensor& vals, bool async){
      ValT* val_ptr = static_cast<ValT*> (vals.data_ptr());
      size_t num_keys = keys.size(0);
      long int* key_ptr = static_cast<long int*> (keys.data_ptr());
      auto ts = worker.PullSample(id, key_ptr, num_keys, val_ptr);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("id"), py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)

    .def("pull_sample", [](WorkerT& worker, ps::SampleID id, py::array_t<long int>& keys, py::array_t<ValT>& vals, bool async){
      auto val_buffer = vals.request();
      ValT* val_ptr = static_cast<ValT*> (val_buffer.ptr);
      auto key_buffer = keys.request();
      size_t num_keys = key_buffer.shape[0];
      long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
      auto ts = worker.PullSample(id, key_ptr, num_keys, val_ptr);
      if (!async) {
        worker.Wait(ts);
      }
      return ts;
    }, py::arg("id"), py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)


    // misc

    .def("begin_setup", [](WorkerT& worker){
      py::gil_scoped_release release;
      worker.BeginSetup();
    })

    .def("end_setup", [](WorkerT& worker){
      py::gil_scoped_release release;
      worker.EndSetup();
    })

    .def("wait_sync", [](WorkerT& worker){
      py::gil_scoped_release release;
      worker.WaitSync();
    })

    .def("wait_replica_sync", [](WorkerT& worker){ // DEPRECATED
      ALOG("[DEPRECATED] You are using worker.wait_replica_sync(). The method has been renamed to worker_wait_sync(). Please switch.");
      py::gil_scoped_release release;
      worker.WaitSync();
    })

    .def("barrier", [](WorkerT& worker){
      py::gil_scoped_release release; // allow for multiple workers to reach barrier()
      worker.Barrier();
    })

    .def("get_key_size", [](WorkerT& worker, long int key_id){
      return worker.GetLen(key_id);
    }, py::arg("key_id") = 0)
    .def_property_readonly("num_keys", &WorkerT::GetNumKeys)
    ;

}
