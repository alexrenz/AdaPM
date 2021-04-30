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


void setup(int num_keys, int num_threads) {
    ps::Postoffice::Get()->enable_dynamic_allocation(num_keys, num_threads);
    ps::Start(0); // start the server (customer id 0)
}

void scheduler(int num_keys, int num_threads) {
  setup(num_keys, num_threads);
  ps::Finalize(0, true);
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup", &setup, "set up lapse");
  m.def("scheduler", &scheduler, "run scheduler");




  py::class_<HandleT>(m, "Handle")
    .def(py::init<long, size_t>())
    ;
  py::class_<ServerT>(m, "Server")
    .def(py::init<int, int>())
    .def(py::init([](torch::Tensor& value_lengths) {
                    size_t num_keys = value_lengths.size(0);
                    auto t = value_lengths.to(torch::kInt64).contiguous(); // torch kInt64 should be a c++ long
                    std::vector<size_t> lengths(t.data_ptr<long>(), t.data_ptr<long>() + t.numel());
                    return new ServerT(num_keys, lengths);
                  }), py::arg("value_lengths").noconvert())
    .def("shutdown", [](ServerT& server){
                       server.shutdown();
                       ps::Finalize(0, true);
                     })
    ;
  py::class_<WorkerT>(m, "Worker")
    .def(py::init([](const int app_id, const int customer_id, ServerT& server) {
                    ps::Start(customer_id);
                    return new WorkerT(0, customer_id, server);
                  }))
    .def("wait", [](WorkerT& worker, int timestamp){ worker.Wait(timestamp); })
    .def("waitall", [](WorkerT& worker){ worker.WaitAll(); })
    .def("finalize", [](WorkerT& worker){ worker.Finalize(); })
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
                   auto do_when_pull_is_done = [&key_ptr, &val_ptr, num_keys](){
                                                 // Do something when this pull is finished
                                                 // Can use any variables that we captured in the [] at the beginning of this lambda
                                                 // E.g.:
                                                 // ALOG("First key: " << key_ptr[0] );
                                        };
                   auto ts = worker.Pull(key_ptr, num_keys,  val_ptr, nullptr, 0, do_when_pull_is_done, async);
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
                   auto do_when_pull_is_done = [&key_ptr, &val_ptr, num_keys](){
                                                 // Do something when this pull is finished
                                                 // Can use any variables that we captured in the [] at the beginning of this lambda
                                                 // E.g.:
                                                 // ALOG("First key: " << key_ptr[0] );
                                        };
                   auto ts = worker.Pull(key_ptr, num_keys,  val_ptr, nullptr, 0, do_when_pull_is_done, async);
                   if (!async) {
                     worker.Wait(ts);
                   }
                   return ts;
    }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)
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
                   auto ts = worker.Push(key_ptr, num_keys, val_ptr, true, async);
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
                   auto ts = worker.Push(key_ptr, num_keys, val_ptr, true, async);
                   if (!async) {
                     worker.Wait(ts);
                   }
                   return ts;
                 }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)
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
                   auto ts = worker.Push(key_ptr, num_keys, val_ptr, false, async);
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
                   auto ts = worker.Push(key_ptr, num_keys, val_ptr, false, async);
                   if (!async) {
                     worker.Wait(ts);
                   }
                   return ts;
                 }, py::arg("keys").noconvert(), py::arg("vals").noconvert(), py::arg("async") = false)
    .def("localize", [](WorkerT& worker, torch::Tensor& keys, bool async){
                   size_t num_keys = keys.size(0);
                   long int* key_ptr = static_cast<long int*> (keys.data_ptr());
                   assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
                   auto ts = worker.Localize(key_ptr, num_keys);
                   if (!async) {
                     worker.Wait(ts);
                   }
                   return ts;
                  }, py::arg("keys").noconvert(), py::arg("async") = false)
    .def("localize", [](WorkerT& worker, py::array_t<long int>& keys, bool async){
                   auto key_buffer = keys.request();
                   size_t num_keys = key_buffer.shape[0];
                   long int* key_ptr = static_cast<long int*> (key_buffer.ptr);
                   assert_keys_in_range(worker.GetNumKeys(), key_ptr, num_keys);
                   auto ts = worker.Localize(key_ptr, num_keys);
                   if (!async) {
                     worker.Wait(ts);
                   }
                   return ts;
                  }, py::arg("keys").noconvert(), py::arg("async") = false)
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
