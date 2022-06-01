/**
 *  Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
#include <thread>
#include <chrono>
#include <cmath>
#include "ps/internal/postoffice.h"
#include "ps/internal/message.h"
#include "ps/base.h"

namespace ps {
Postoffice::Postoffice() {
  van_ = Van::Create("zmq");
  env_ref_ = Environment::_GetSharedRef();
}

void Postoffice::InitEnvironment() {
  const char* val = NULL;
  num_workers_ = 0;
  val =  CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_SERVER"));
  num_servers_ = atoi(val);
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  std::string role(val);
  is_worker_ = role == "worker";
  is_server_ = role == "server";
  is_scheduler_ = role == "scheduler";
  verbose_ = GetEnv("PS_VERBOSE", 0);
}

void Postoffice::Start(int customer_id, const char* argv0, const bool do_barrier) {
  start_mu_.lock();
  if (init_stage_ == 0) {
    InitEnvironment();
    // init glog
    if (argv0) {
      dmlc::InitLogging(argv0);
    } else {
      dmlc::InitLogging("ps\0");
    }

    // init node info. // note: when co-locating, num_workers_=0
    for (int i = 0; i < num_workers_; ++i) {
      int id = WorkerRankToID(i);
      for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
                    kWorkerGroup + kScheduler,
                    kWorkerGroup + kServerGroup + kScheduler}) {
        node_ids_[g].push_back(id);
      }
    }

    for (int i = 0; i < num_servers_; ++i) {
      int id = ServerRankToID(i);
      for (int g : {id, kServerGroup, kWorkerGroup + kServerGroup,
                    kServerGroup + kScheduler,
                    kWorkerGroup + kServerGroup + kScheduler}) {
        node_ids_[g].push_back(id);
      }

      // for co-locating, we have worker threads within server processes
      // thus, to have a barrier over all worker threads,
      // we create a new kWorkerThreadGroup [sysChange]
      for(uint j=0; j!=num_worker_threads(); ++j) {
        node_ids_[kWorkerThreadGroup].push_back(id);
      }
    }

    for (int g : {kScheduler, kScheduler + kServerGroup + kWorkerGroup,
                  kScheduler + kWorkerGroup, kScheduler + kServerGroup}) {
      node_ids_[g].push_back(kScheduler);
    }
    init_stage_++;
  }
  start_mu_.unlock();

  // start van
  van_->Start(customer_id, num_network_threads_); // [sysChange]

  start_mu_.lock();
  if (init_stage_ == 1) {
    // record start time
    start_time_ = time(NULL);
    init_stage_++;
  }
  start_mu_.unlock();
  // do a barrier here (make sure all nodes are up)
  if (is_primary_ps(customer_id) || is_scheduler_) { // if we co-locate workers and servers, only the (primary) server thread participates in the barrier [sysChange]
    if (do_barrier) Barrier(customer_id, kWorkerGroup + kServerGroup + kScheduler);
  }
}

void Postoffice::Finalize(const int customer_id, const bool do_barrier) {
  if (do_barrier) Barrier(customer_id, kWorkerGroup + kServerGroup + kScheduler);
  if (is_primary_ps(customer_id)) {
    num_workers_ = 0;
    num_servers_ = 0;
    van_->Stop();
    init_stage_ = 0;
    customers_.resize(0);
    node_ids_.clear();
    barrier_done_.clear();
    server_key_ranges_.clear();
    heartbeats_.clear();
    if (exit_callback_) exit_callback_();
  }
}


void Postoffice::AddCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  // check if the customer id has existed
  int customer_id = CHECK_NOTNULL(customer)->customer_id();
  CHECK_EQ(customers_[customer_id], nullptr) << "customer_id " \
    << customer_id << " already exists\n";
  customers_[customer_id] = customer;
  std::unique_lock<std::mutex> ulk(barrier_mu_);
  for(int node_group=0; node_group!=8; ++node_group) {
    barrier_done_[node_group].insert(std::make_pair(customer_id, false));
  }
}


void Postoffice::RemoveCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  int customer_id = CHECK_NOTNULL(customer)->customer_id();
  customers_[customer_id] = nullptr;
}


Customer* Postoffice::GetCustomer(int app_id, int customer_id, int timeout) const {
  // fast (normal) mode
  if (customers_[customer_id] != nullptr) {
    return customers_[customer_id];
  }


  // in rare cases, the customer might not be set up yet. wait for that
  for (int i = 0; i < timeout * 1000 + 1; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (customers_[customer_id] != nullptr) {
      return customers_[customer_id];
    }
  }

  ALOG("Failed to find customer " << customer_id << " within timeout " << timeout);
  return nullptr;
}

void Postoffice::Barrier(int customer_id, int node_group) {
  if (GetNodeIDs(node_group).size() <= 1) return;
  auto role = van_->my_node().role;
  if (role == Node::SCHEDULER) {
    CHECK(node_group & kScheduler);
  } else if (role == Node::WORKER) {
    CHECK(node_group & kWorkerGroup);
  } else if (role == Node::SERVER) {
    CHECK(node_group & kServerGroup || node_group == kWorkerThreadGroup); // allow worker thread barriers [sysChange]
  }

  std::unique_lock<std::mutex> ulk(barrier_mu_);
  barrier_done_[node_group][customer_id] = false;
  Message req;
  req.meta.recver = kScheduler;
  req.meta.request = true;
  req.meta.control.cmd = Control::BARRIER;
  req.meta.app_id = 0;
  req.meta.customer_id = customer_id;
  req.meta.control.barrier_group = node_group;
  req.meta.timestamp = van_->GetTimestamp();
  CHECK_GT(van_->Send(req), 0);
  barrier_cond_.wait(ulk, [this, customer_id, node_group] {
      return barrier_done_[node_group][customer_id];
    });
}

const std::vector<Range>& Postoffice::GetServerKeyRanges() {
  server_key_ranges_mu_.lock();
  if (server_key_ranges_.empty()) {
    for (int i = 0; i < num_servers_; ++i) {
      server_key_ranges_.push_back(Range(
          std::ceil(1.0 * num_keys_ / num_servers_) * i,    // [sysChange] guarantee that ranges include max. key
          std::ceil(1.0 * num_keys_ / num_servers_) * (i+1)));
    }
  }
  server_key_ranges_mu_.unlock();
  return server_key_ranges_;
}

void Postoffice::Manage(const Message& recv) {
  CHECK(!recv.meta.control.empty());
  const auto& ctrl = recv.meta.control;
  if (ctrl.cmd == Control::BARRIER && !recv.meta.request) {
    barrier_mu_.lock();
    for (uint customer_id = 0; customer_id < barrier_done_[ctrl.barrier_group].size(); ++customer_id) {
      barrier_done_[ctrl.barrier_group][customer_id] = true;
    }
    barrier_mu_.unlock();
    barrier_cond_.notify_all();
  }
}

std::vector<int> Postoffice::GetDeadNodes(int t) {
  std::vector<int> dead_nodes;
  if (!van_->IsReady() || t == 0) return dead_nodes;

  time_t curr_time = time(NULL);
  const auto& nodes = is_scheduler_
    ? GetNodeIDs(kWorkerGroup + kServerGroup)
    : GetNodeIDs(kScheduler);
  {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    for (int r : nodes) {
      auto it = heartbeats_.find(r);
      if ((it == heartbeats_.end() || it->second + t < curr_time)
            && start_time_ + t < curr_time) {
        dead_nodes.push_back(r);
      }
    }
  }
  return dead_nodes;
}
}  // namespace ps
