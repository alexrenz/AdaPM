/**
 *  Copyright (c) 2018 by Contributors
 */
#ifndef PS_COLOC_SERVER_TRANSFERS_
#define PS_COLOC_SERVER_TRANSFERS_
#include <algorithm>
#include <utility>
#include <vector>
#include "ps/coloc_kv_worker.h"
#include "ps/coloc_kv_server.h"
#include <valarray>
#include <thread>
#include <mutex>
#include <memory>
#include <iostream>
#include <sstream>

namespace ps {


  /**
     \brief Manages information about ongoing transfers, i.e., about parameters
            that are currently in the process of being moved to the local PS

            Thread safety is managed by the PS locks (see server handle)
  */
  template <typename Val>
    class ColoServerTransfers {
  public:
  ColoServerTransfers(size_t num_keys): transfers(num_keys) {}

    /**
     * \brief Queues a local push or pull request for a parameter that is in transfer.
     *        The request will be answered locally when the transfer is finished.
     *
     *        This method is run by the worker thread that issued the push/pull request
     *        It does not acquire the necessary lock for the in-transfer data structure
     */
    void queueLocalRequestUnsafe(const Key key, const int ts, Customer* customer) {
      // mark the calling thread as waiting (it is notified on finish)
      transfers[key].threads.push_back(WaitingThread(customer, ts));
    }

    /**
     * \brief Queue a remote push or pull request for a parameter that is in transfer.
     *        The respective (queued) message will be sent out when all parameters of
     *        the message are answered.
     */
    void queueRemoteRequestUnsafe(const Key key, std::shared_ptr<QueuedMessage<Val>>& queued_msg) {
      queued_msg->kvs.keys.push_back(key);
      transfers[key].messages.push_back(queued_msg);
    }

    /**
     * \brief Adds a push to the queue of a transfer. When the transfer is finished,
     *        the push will be processed in order.
     */
    void registerPushOp(const Key key, Val* val) {
      transfers[key].ops.push_back(std::make_pair(true, val));
    }

    /**
     * \brief Adds a remote push to the queue of a transfer. Makes sure that the
     *        data that shall be pushed will still be around when the transfer finishes.
     */
    void addRemotePushToQueueUnsafe(const Key key, Val* val, std::shared_ptr<KVPairs<Val>>& data_ptr) {
      // ensure that the data of the request is still available when the transfer is finished
      transfers[key].push_data_ptrs.push_back(data_ptr);

      // add to queue
      registerPushOp(key, val);
    }

    /**
     * \brief Adds a remote push to the queue of a transfer. Makes sure that the
     *        data will still be around once the transfer finishes.
     */
    void addLocalPushToQueueUnsafe(const Key key, Val* val, const size_t len) {
      // make a copy of the value that is being pushed and keep the copy around
      //  (This is necessary because we don't require the application to keep
      //   the value vector around after the Push()-call has returned)
      std::vector<Val> push_copy (val, val+len);
      Val* store_val = push_copy.data();
      transfers[key].pushs.push_back(std::move(push_copy));

      // add to queue
      registerPushOp(key, store_val);
    }

    /**
     * \brief Adds a memory location to a transfer. When the transfer is finished,
     *        the current value will be copied to this location.
     */
    void addPullToQueueUnsafe(const Key key, Val* val_loc) {
      // store the target location of pulls
      // we copy the latest value there when the transfer finishes
      transfers[key].ops.push_back(std::make_pair(false, val_loc));
    }

    /**
     * \brief Returns whether a key is currently in the process of being localized
     */
    inline bool isInTransferUnsafe(Key key) {
      return transfers[key].ongoing();
    }

    /**
     * \brief Note down a subsequent localize for a parameter that is in transfer right now.
     *        This node will directly forward the parameter to the subsequent localize node
     *        as soon as its own localize finishes.
     *
     *        Returns true if the note was successful, false if this parameter is not in
     *        transfer right now.
     *
     *        Note: this method is run by the parameter server thread.
     */
    bool noteSubsequentLocalizeUnsafe(Key key, std::shared_ptr<QueuedMessage<Val>>& queued_msg) {
      if (transfers[key].ongoing()) {
        CHECK(!transfers[key].subsequentLocalize) << "FATAL! Already have a subsequent localize for key " << key << " at " << Postoffice::Get()->my_rank();

        transfers[key].subsequentLocalize = true;

        // make sure meta is still available when we forward the parameter
        transfers[key].subsequentLocalizeMsg = queued_msg;
        return true;
      } else {
        return false;
      }
    }

    /**
     * \brief Start the bookkeeping for the localization of a key. (we create a data structure
     *        that queues all requests that arrive until this node receives the key)
     *
     *        Note: this method is run by the worker thread that requests the localize!
     *        Note: this method is not thread safe
     */
    void startTransferUnsafe(Key key) {
      CHECK(!transfers[key].ongoing()) << "FATAL! Starting a localize for key " << key << ", but already have an ongoing localize";
      transfers[key].start();
    }

    ps::Transfer<Val>& getTransferUnsafe(Key key) {
      return transfers[key];
    }

  private:
    /* Holds information about the parameters that are "in transfer" to the local PS at the moment */
    std::vector<Transfer<Val>> transfers;
  };

}  // namespace ps
#endif  // PS_COLOC_SERVER_TRANSFERS_
