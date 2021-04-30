/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_THREADSAFE_QUEUE_H_
#define PS_INTERNAL_THREADSAFE_QUEUE_H_
#include <deque>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "ps/base.h"
namespace ps {

/**
 * \brief thread-safe queue allowing push and waited pop
 *
 *        [sysChange] change data structure to a deque and allow for front placements
 */
template<typename T> class ThreadsafeQueue {
 public:
  ThreadsafeQueue() { }
  ~ThreadsafeQueue() { }

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void Push(T new_value, bool placeFront=false) {
    mu_.lock();
    if (placeFront) {
      queue_.push_front(std::move(new_value));
    } else {
      queue_.push_back(std::move(new_value));
    }
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  size_t WaitAndPop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this]{return !queue_.empty();});
    *value = std::move(queue_.front());
    queue_.pop_front();
    return queue_.size();
  }

 private:
  mutable std::mutex mu_;
  std::deque<T> queue_;
  std::condition_variable cond_;
};

}  // namespace ps

// bool TryPop(T& value) {
//   std::lock_guard<std::mutex> lk(mut);
//   if(data_queue.empty())
//     return false;
//   value=std::move(data_queue.front());
//   data_queue.pop();
//   return true;
// }
#endif  // PS_INTERNAL_THREADSAFE_QUEUE_H_
