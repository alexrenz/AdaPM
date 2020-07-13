/**
 *  Copyright (c) 2018 by Contributors
 */
#ifndef PS_COLOC_KV_WORKER_H_
#define PS_COLOC_KV_WORKER_H_
#include <algorithm>
#include <utility>
#include <vector>
#include "ps/kv_app.h"
#include "ps/addressbook.h"

#include <iostream>
#include <iomanip>
#include <sstream>

namespace ps {

  // forward declaration
  template <typename Val, typename Handle>
    class ColoKVServer;

  /**
   * \brief A worker node that can \ref Push (\ref Pull) key-value pairs to (from) server
   * nodes, with support for local communication (within process)
   *
   * \tparam Val the type of value, which should be primitive types such as
   * int32_t and float
   */
  template<typename Val, typename Handle>
    class ColoKVWorker : public KVWorker<Val> {
  public:
    /** avoid too many this-> */
    using SimpleApp::obj_;
    /**
     * \brief callback function for \ref Push and \ref Pull
     *
     * It is called by the data receiving thread of this instance when the push or
     * pull is actually finished. Namely the kv pairs have already written into
     * servers' data structure or the kv pairs have already pulled back.
     */
    using Callback = std::function<void()>;

    /**
     * \brief constructor
     *
     * \param app_id the app id, should match with \ref KVServer's id
     * \param customer_id the customer id which is unique locally
     */
    explicit ColoKVWorker(const int app_id, const int customer_id, ColoKVServer<Val, Handle>& sserver) : KVWorker<Val>(), server(sserver), customer_id_(customer_id) {
      using namespace std::placeholders;
      this->obj_ = new Customer(app_id, customer_id, std::bind(&ColoKVWorker<Val, Handle>::Process, this, _1));
      // note: obj_ is deleted by the destructor of KVWorker
    }

    // debug statistics
    ~ColoKVWorker() {
      server.num_pull_ops += num_pull_ops;
      server.num_pull_ops_local += num_pull_ops_local;
      server.num_push_ops += num_push_ops;
      server.num_push_ops_local += num_push_ops_local;
      server.num_pull_params += num_pull_params;
      server.num_pull_params_local += num_pull_params_local;
      server.num_push_params += num_push_params;
      server.num_push_params_local += num_push_params_local;
    }

    // reset re-used data structures
    inline void reset(size_t size) {
      local.resize(size);
      std::fill_n(local.begin(), size, 0);
    }

    /**
     * \brief Push key--values into the PS (overrides KVWorker::Push())
     *
     * If shared memory access is enabled, this method attempts to push all keys locally.
     * If this succeeds (i.e., if all keys are in the local PS),
     * method returns -1 (meaning "no need to wait for a request to finish").
     * Otherwise, it returns a timestamp to wait for.
     *
     * @param keys a list of keys
     * @param vals the according values
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Push(const std::vector<Key>& keys,
             std::vector<Val>& vals) {
      size_t vpk = server.request_handle_.get_vpk();
      ++num_push_ops;
      num_push_params += keys.size();

      // STEP 1: fast special case (if all parameters are local)
      reset(keys.size());
      size_t numLocal = 0;
      if (Postoffice::Get()->shared_memory_access()) {
        for (size_t i = 0; i < keys.size(); ++i) {
          // try to process the request locally
          Key key = keys[i];
          local[i] = server.request_handle_.attemptLocalPush(key, &vals[i*vpk]);
          numLocal += local[i];
        }
      }
      num_push_params_local += numLocal;
      if (numLocal == keys.size()) {
        ++num_push_ops_local;
        return -1; // fast case done
      }


      // STEP 2: slow general case (send out requests to other servers)
      int ts = obj_->NewRequest(kServerGroup, keys.size());
      KVPairs<Val> kvs;
      kvs.keys = SArray<Key>(keys);
      kvs.vals = SArray<Val>(vals);
      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers()); // TODO-perf: reuse

      // parse the non-processed keys a second time
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!local[i]) {
          Key key = keys[i];
          ps::Status status = server.request_handle_.processPush(key, &vals[i*vpk], ts, obj_);
          if (status == ps::Status::LOCAL) {
            ++numLocal;
          } else if (status == ps::Status::REMOTE) {
            auto destination = server.addressbook.getDirections(key, true);
            requests[destination].keys.push_back(key);
            std::copy_n(kvs.vals.begin()+i*vpk, vpk, std::back_inserter(requests[destination].vals));
          }
        }
      }

      SendRequests(ts, true, requests, numLocal, 0);
      return ts;
    }

    /**
     * \brief Pull key--values from the PS (overwrites KVWorker::Pull())
     *
     * If shared memory access is enabled, this method attempts to pull all keys locally.
     * If this succeeds (i.e., if all keys are in the local PS),
     * method returns -1 (meaning "no need to wait for a request to finish").
     * Otherwise, it returns a timestamp to wait for.
     *
     * @param keys a list of keys
     * @param vals a vector to store the vals into
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Pull(const std::vector<Key>& keys,
             std::vector<Val>* vals,
             SArray<int>* lens = nullptr,
             int cmd = 0,
             const Callback& cb = nullptr) {
      auto vpk = server.request_handle_.get_vpk();
      ++num_pull_ops;
      num_pull_params += keys.size();

      // STEP 1: fast special case (if all parameters are local)
      size_t numLocal = 0;
      reset(keys.size());
      if (Postoffice::Get()->shared_memory_access()) {
        for (size_t i = 0; i < keys.size(); ++i) {
          // try to process the request locally
          Key key = keys[i];
          local[i] = server.request_handle_.attemptLocalPull(key, vals->data()+i*vpk);
          numLocal += local[i];
        }
      }
      num_pull_params_local += numLocal;
      if (numLocal == keys.size()) {
        ++num_pull_ops_local;
        return -1;
      }

      // Step 2: slow general case

      // create request
      int ts = obj_->NewRequest(kServerGroup, keys.size());
      orig_kv_mu_.lock();
      request_original_kvs_[ts] = std::make_pair(&keys, vals);
      orig_kv_mu_.unlock();

      this->AddCallback(ts, [this, ts, keys, vals, lens, cb]() mutable {
          orig_kv_mu_.lock();
          request_original_kvs_.erase(ts);
          orig_kv_mu_.unlock();
          if (cb) cb();
        });
      KVPairs<Val> kvs;
      kvs.keys = SArray<Key>(keys);
      kvs.vals = SArray<Val>(vals->data(), 0);

      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers()); // TODO: reuse

      // parse the non-processed keys a second time
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!local[i]) {
          Key key = keys[i];
          ps::Status status = server.request_handle_.processPull(key, vals->data()+i*vpk, ts, obj_);
          if (status == ps::Status::LOCAL) {
            ++numLocal;
          } else if (status == ps::Status::REMOTE) {
            auto destination = server.addressbook.getDirections(key, true);
            requests[destination].keys.push_back(key);
          }
        }
      }

      // send out messages and return timestamp
      SendRequests(ts, false, requests, numLocal, 0);
      return ts;
    }

    /**
     * \brief Pull the value of a parameter if the parameter is local
     *        Returns true and stores value in *val if parameter is local, returns false otherwise
     */
    int PullIfLocal(const Key key, std::vector<Val>* vals) {
      return server.request_handle_.attemptLocalPull(key, vals->data());
    }

    /**
     * \brief Make parameters local to this parameter server
     *
     * @param keys the list of keys to localize
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Localize(const std::vector<Key>& keys) {

      int ts = obj_->NewRequest(kServerGroup, keys.size());
      KVPairs<Val> kvs;
      kvs.keys = SArray<Key>(keys);

      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers()); // TODO: reuse

      size_t numLocal = 0;
      for (size_t i = 0; i < keys.size(); ++i) {
        Key key = keys[i];
        ps::Status status = server.request_handle_.processLocalize(key, ts, obj_);
        if (status == ps::Status::LOCAL) {
          ++numLocal;
        } else if (status == ps::Status::REMOTE) {
          auto destination = -1;
          if (server.addressbook.isManagedHere(key)) {
            // get the old owner and set new owner atomically
            auto old_owner = server.addressbook.updateResidence(key, Postoffice::Get()->my_rank());
            destination = old_owner;
            TLOG(key, Postoffice::Get()->my_rank(), "self manager. update residence and forward to " << destination);
          } else {
            // if this is a localize request and this not is not the manager, we send a request to the manager of the key
            destination = server.addressbook.getManager(key);
            TLOG(key, Postoffice::Get()->my_rank(), "start. send to manager " << destination);
          }
          // for localize: special treatment
          requests[destination].keys.push_back(key);
        }
      }

      // send out messages and return timestamp
      SendRequests(ts, false, requests, numLocal, Control::LOCALIZE);
      return ts;
    }

    /**
     * \brief Wait for a timestamp (overrides KVWorker::Wait())
     */
    void Wait(int timestamp) {
      if(timestamp == -1) // no need to wait. the corresponding call was answered locally
        return;
      else
        obj_->WaitRequest(timestamp);
    }

    /**
     * \brief Wait for 2 timestamps
     */
    void Wait(int ts1, int ts2) {
      if(ts1 != -1) obj_->WaitRequest(ts1);
      if(ts2 != -1) obj_->WaitRequest(ts2);
    }

    /**
     * \brief Wait for 3 timestamps
     */
    void Wait(int ts1, int ts2, int ts3) {
      if(ts1 != -1) obj_->WaitRequest(ts1);
      if(ts2 != -1) obj_->WaitRequest(ts2);
      if(ts3 != -1) obj_->WaitRequest(ts3);
    }

    /**
     * \brief Wait for all requests issued by this worker
     */
    void WaitAll() {
      int num_requests = obj_->num_requests();
      for(int ts=0; ts!=num_requests; ++ts) {
        obj_->WaitRequest(ts);
      }
    }

    /**
     * \brief Check whether a particular request is finished
     */
    bool IsFinished(int ts) {
      return ts == -1 || obj_->HasAllResponses(ts);
    }

    /**
     * \brief Synchronize value caches
     */
    int SynchronizeValueCaches() {
      ADLOG("[s" << Postoffice::Get()->my_rank() << "] Synchronize value caches");
      // clear caches and collect all cached parameter updates
      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers());
      auto num_keys = server.request_handle_.clearValueCachesAndCollectUpdates(requests, server.addressbook);

      // send the cached updates to the servers
      int ts = obj_->NewRequest(kServerGroup, num_keys);
      SendRequests(ts, true, requests, 0, 0);
      return ts;
    }

    /**
     * \brief Clear value caches
     */
    int InvalidateValueCaches() {
      ADLOG("[s" << Postoffice::Get()->my_rank() << "] Invalidate value caches");
      // clear all value caches
      std::vector<KVPairs<Val>> requests (0);
      server.request_handle_.clearValueCachesAndCollectUpdates(requests, server.addressbook, false);
      return -1;
    }

    /**
     * \brief Reset locality statistics. Useful if one does not want to measure locality of parameter initialization
     */
    void ResetStats() {
      num_pull_ops = 0;
      num_pull_ops_local = 0;
      num_push_ops = 0;
      num_push_ops_local = 0;
      num_pull_params = 0;
      num_pull_params_local = 0;
      num_push_params = 0;
      num_push_params_local = 0;
    }

    /**
     * \brief Wait for a barrier among all servers
     */
    void Barrier() {
      Postoffice::Get()->Barrier(customer_id_, kWorkerThreadGroup);
    }

  private:

    /**
     * \brief Send requests to other parameter servers to answer push/pull/localize calls
     */
    void SendRequests(const int timestamp, const bool push, const std::vector<KVPairs<Val>>& requests,
                      const int numLocal, int cmd=0) {
      if (numLocal != 0) { // add response count for the requests that were answered already
        obj_->AddResponse(timestamp, numLocal);
      }
      // untypical case: have all responses already -> run callback right now
      if (obj_->HasAllResponses(timestamp)) {
        this->RunCallback(timestamp);
      }

      // send out messages to other servers
      for (unsigned int i=0; i!=requests.size(); ++i) {
        if (!requests[i].keys.empty()) {
          this->Request(i, requests[i], push, cmd, timestamp);
        }
      }
    }

    /**
     * \brief Send out one request to a specific parameter server
     */
    void Request(const int destination, const KVPairs<Val>& kvs, const bool push,
                 const int cmd, const int timestamp) {
      // Message object
      Message msg;
      msg.meta.app_id = obj_->app_id();
      msg.meta.customer_id = obj_->customer_id();
      msg.meta.request     = true;
      msg.meta.push        = push;
      msg.meta.head        = cmd;
      msg.meta.timestamp   = timestamp;
      msg.meta.recver      = Postoffice::Get()->ServerRankToID(destination);
      if (kvs.keys.size()) {
        msg.AddData(kvs.keys);
        msg.AddData(kvs.vals);
        if (kvs.lens.size()) {
          msg.AddData(kvs.lens);
        }
      }

      // Debug output
      if (Postoffice::Get()->shared_memory_access() && msg.meta.recver == Postoffice::Get()->ServerRankToID(Postoffice::Get()->my_rank())) {
        Key key = kvs.keys[0];
        bool isloc = server.request_handle_.isLocal(key);
        bool ispip = server.request_handle_.isInTransfer(key);
        ADLOG("Customer r" << Postoffice::Get()->my_rank() << "c" << obj_->customer_id() << " sends a remote message to its local PS for ts " << timestamp << ", " << (isLocalize(msg.meta.head) ? "localize" : (msg.meta.push ? "push" : "pull")) << " " << (msg.meta.request ? "request" : "response") << ". For key " << key << " (" << kvs.keys.size() << " total). local: " << isloc << ", pip:" << ispip << ", manager: " << server.addressbook.getManager(key) << ", destination: " << destination << ", me: " << Postoffice::Get()->my_rank() << ", directions: " << server.addressbook.getDirections(key, true));
      }

      // Send message
      Postoffice::Get()->van()->Send(msg);
    }

    /**
     * \brief Process a response for a pull or push request (i.e., receive handle)
     *
     *        We note that the message has arrived. For pulls, we additionally
     *        store the received values into the original value array
     */
    bool Process(const Message& msg);

    /** \brief reference to the parameter server of this process */
    ColoKVServer<Val, Handle>& server;

    // pointers to original keys and values (we use this for pushs: when a message arrives, we put the arriving values directly into the original array)
    std::unordered_map<int, std::pair<const std::vector<Key>*, std::vector<Val>*>> request_original_kvs_;
    std::mutex orig_kv_mu_;

    // The customer id of the thread that owns this KVWorker instance (used for barrier)
    const int customer_id_;

    // Note status (local/not local) of parameters of a request
    std::vector<bool> local {};

    long num_pull_ops=0, num_pull_ops_local=0, num_push_ops=0, num_push_ops_local=0;
    long num_pull_params=0, num_pull_params_local=0, num_push_params=0, num_push_params_local=0;
  };



  template <typename Val, typename Handle>
    bool ColoKVWorker<Val, Handle>::Process(const Message& msg) {
    // store the data for pulling
    int ts = msg.meta.timestamp;

    if(msg.meta.control.cmd != Control::WAKE_SIGNAL) {
      if (!msg.meta.push && msg.data.size()) { // pull call
        CHECK_GE(msg.data.size(), (size_t)2);
        KVPairs<Val> kvs;
        kvs.keys = msg.data[0];
        kvs.vals = msg.data[1];
        if (msg.data.size() > (size_t)2) {kvs.lens = msg.data[2];}

        // get reference to original keys + val arrays of this timestamp
        orig_kv_mu_.lock();
        auto orig = request_original_kvs_[ts];
        orig_kv_mu_.unlock();

        CHECK(orig.first) << "FATAL: have no place to put reply to a pull requests at worker " << Postoffice::Get()->my_rank() << "::" << obj_->customer_id() << " for ts " << ts <<". Pointer is " << orig.first;

        // go trough requested keys and match with the keys of this message,
        // place received vals at the correct positions of the array
        // note: we can do this without lock because (a) there is only one receive thread
        //    (b) if there were multiple, there is separated write areas in the val array

        size_t pos = 0;
        auto &requested_keys = *orig.first;
        auto vpk = server.request_handle_.get_vpk();
        auto senderRank = Postoffice::Get()->IDtoRank(msg.meta.sender);
        for (size_t i = 0; i != requested_keys.size(); ++i) {
          Key key = kvs.keys[pos];
          if (requested_keys[i] == key) {
            // write vals to correct position
            std::copy_n(kvs.vals.begin()+pos*vpk, vpk, (*orig.second).begin()+i*vpk);

            // update location caches
            if (Postoffice::Get()->use_location_caches()) {
              server.addressbook.updateCache(key, senderRank);
            }

            // update value caches
            if (Postoffice::Get()->use_value_caches()) {
              server.request_handle_.updateValueCache(key, kvs.vals.begin());
            }

            // move pointer forward
            ++pos;

            // early stop: stop when all keys of this response are processed
            if (pos == kvs.keys.size()) {
                break;
            }
          }
        }

        // increase receive count
        obj_->AddResponse(ts, kvs.keys.size());
      } else if(msg.meta.push && msg.data.size()) { // push call // TODO could update caches if we send back exactly the keys that were processed
        // responses to push messages return the number of keys that were updated successfully by the sending server
        // we use this number to determine when all pushs have gone through

        auto resp_keys = SArray<Key>(msg.data[0]);
        Key num_confirmed_keys = resp_keys[0];
        obj_->AddResponse(ts, num_confirmed_keys);
      } else {
        LL << "ERROR: Pull/Push response without data";
      }
    } else { /* only run callback (below) if this is a WAKE_SIGNAL message */ }

    // finished, run callbacks
    if (obj_->HasAllResponses(ts)) {
      this->RunCallback(ts);
    }
    return false;
  }
}  // namespace ps
#endif  // PS_COLOC_KV_WORKER_H_
