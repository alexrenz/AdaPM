/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/customer.h"
#include "ps/internal/postoffice.h"
#include <iomanip>
namespace ps {

const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();

Customer::Customer(int app_id, int customer_id, const Customer::RecvHandle& recv_handle)
    : app_id_(app_id), customer_id_(customer_id), recv_handle_(recv_handle) {
  Postoffice::Get()->AddCustomer(this);
  recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
  std::string name = std::to_string(Postoffice::Get()->my_rank()) + "-customer-" + std::to_string(customer_id);
  if (customer_id == 0) name = std::to_string(Postoffice::Get()->my_rank()) + "-ps";
  SET_THREAD_NAME(recv_thread_, name.c_str());
}

Customer::~Customer() {
  Postoffice::Get()->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  recv_thread_->join();
}

int Customer::NewRequest(int recver, int num) { // default for num is -1
  std::lock_guard<std::mutex> lk(tracker_mu_);
  if (num == -1) {
    num = Postoffice::Get()->GetNodeIDs(recver).size();
  }
  tracker_.push_back(std::make_pair(num, 0));
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp].first == tracker_[timestamp].second;
    });
}

int Customer::NumResponse(int timestamp) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp].second;
}

int Customer::HasAllResponses(int timestamp) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  // ADLOG("r" << Postoffice::Get()->my_rank() << ":c" << customer_id_ << ":t" << timestamp <<" has " << tracker_[timestamp].second << "/" << tracker_[timestamp].first << " responses");
  return tracker_[timestamp].first == tracker_[timestamp].second;
}

bool Customer::AddResponse(int timestamp, int num) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  tracker_[timestamp].second += num;

  return tracker_[timestamp].first == tracker_[timestamp].second;
}

void Customer::Receiving() {
  // stats
  long long q_size = 0;
  long long iterations = 0;
  auto r = Postoffice::Get()->my_rank();

  // receive loop
  while (true) {
    Message recv;
    ++iterations;
    q_size += recv_queue_.WaitAndPop(&recv);
    if (!recv.meta.control.empty() &&
        recv.meta.control.cmd == Control::TERMINATE) {
      if (customer_id_ == 0) {
        ADLOG("Mean length of recv queue in ps-" << r << ": " << std::setprecision(5) << 1.0*q_size/iterations);
      }
      break;
    }
    bool count_msg = recv_handle_(recv);
    if (!recv.meta.request) {
      std::lock_guard<std::mutex> lk(tracker_mu_);
      if(count_msg) tracker_[recv.meta.timestamp].second++;
      tracker_cond_.notify_all();
    }
  }
}

}  // namespace ps
