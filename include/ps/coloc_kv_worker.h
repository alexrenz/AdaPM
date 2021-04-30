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

      if(server.sampling_ != nullptr) {
        server.sampling_->registerWorker(customer_id_, this);
      }
    }

    // debug statistics
    ~ColoKVWorker() {
      server.num_pull_ops += num_pull_ops;
      server.num_pull_ops_local += num_pull_ops_local;
      server.num_push_ops += num_push_ops;
      server.num_push_ops_local += num_push_ops_local;
      server.num_pull_params += num_pull_params;
      server.num_pull_params_local += num_pull_params_local;
      server.num_pull_params_in_transfer += num_pull_params_in_transfer;
      server.num_push_params += num_push_params;
      server.num_push_params_local += num_push_params_local;
      server.num_push_params_in_transfer += num_push_params_in_transfer;
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
      ++num_push_ops;
      num_push_params += keys.size();

      size_t len = 0, pos = 0;

      // STEP 1: fast special case (if all parameters are local)
      reset(keys.size());
      size_t numLocal = 0;
      if (Postoffice::Get()->shared_memory_access()) {
        for (size_t i = 0; i < keys.size(); ++i) {
          // try to process the request locally
          Key key = keys[i];
          CHECK(key < Postoffice::Get()->max_key())<< "[ERROR] Push key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->max_key() << ")";
          len = server.request_handle_.get_len(key);
          local[i] = server.request_handle_.attemptLocalPush(key, vals.data() + pos);
          pos += len;
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
      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers());
      std::vector<size_t> requests_num_keys (Postoffice::Get()->num_servers());
      std::vector<size_t> requests_len_vals (Postoffice::Get()->num_servers());
      std::vector<int> destinations (keys.size());
      pos = 0; // reset for 2nd pass

      // determine which keys are remote and to where we need to send the messages
      for (size_t i = 0; i < keys.size(); ++i) {

        Key key = keys[i];
        len = server.request_handle_.get_len(key);
        if (!local[i]) {
          ps::Status status = server.request_handle_.processPush(key, &vals[pos], ts, obj_);
          if (status == ps::Status::LOCAL) {
            local[i] = true;
            ++numLocal;
          } else if (status == ps::Status::IN_TRANSFER) {
            local[i] = true;  // parameter is already on its way to this node, no need to send a message
            ++num_push_params_in_transfer;
          } else { // remote
            destinations[i] = server.addressbook.getDirections(key, true);
            requests_num_keys[destinations[i]] += 1;
            requests_len_vals[destinations[i]] += len;
          }
        }
        pos += len;
      }

      // allocate memory for keys and values in the requests
      for (size_t i = 0; i < requests.size(); ++i) {
        requests[i].keys.reserve(requests_num_keys[i]);
        requests[i].vals.reserve(requests_len_vals[i]);
      }

      // copy keys and values to requests
      pos = 0;
      for (size_t i = 0; i < keys.size(); ++i) {
        Key key = keys[i];
        len = server.request_handle_.get_len(key);
        if (!local[i]) {
          requests[destinations[i]].keys.push_back(key);
          std::copy_n(vals.begin() + pos, len, std::back_inserter(requests[destinations[i]].vals));
        }
        pos += len;
      }

      // send out requests
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
      ++num_pull_ops;
      num_pull_params += keys.size();

      size_t len = 0, pos = 0;

      // STEP 1: fast special case (if all parameters are local)
      size_t numLocal = 0;
      reset(keys.size());
      if (Postoffice::Get()->shared_memory_access()) {
        for (size_t i = 0; i < keys.size(); ++i) {
          // try to process the request locally
          Key key = keys[i];

          CHECK(key < Postoffice::Get()->max_key())<< "[ERROR] Pull key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->max_key() << ")";

          len = server.request_handle_.get_len(key);
          local[i] = server.request_handle_.attemptLocalPull(key, vals->data() + pos, true, true);

          pos += len;
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

      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers()); // TODO: reuse

      pos = 0; // reset for 2nd pass
      // parse the non-processed keys a second time
      for (size_t i = 0; i < keys.size(); ++i) {
        Key key = keys[i];
        len = server.request_handle_.get_len(key);

        if (!local[i]) {
          ps::Status status = server.request_handle_.processPull(key, vals->data() + pos, ts, obj_);
          if (status == ps::Status::LOCAL) {
            ++numLocal;
          } else if (status == ps::Status::IN_TRANSFER) {
            ++num_pull_params_in_transfer;
          } else { // remote
            auto destination = server.addressbook.getDirections(key, true);
            requests[destination].keys.push_back(key);
          }
        }
        pos += len;
      }

      // send out messages and return timestamp
      SendRequests(ts, false, requests, numLocal, 0);
      return ts;
    }

    /**
     * \brief Pull the value of a parameter if the parameter is local
     *        Returns true and stores value in *val if parameter is local, returns false otherwise
     */
    bool PullIfLocal(const Key key, std::vector<Val>* vals) {
      CHECK(key < Postoffice::Get()->max_key())<< "[ERROR] Pull key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->max_key() << ")";

      // try to check locality without acquiring a lock
      if (server.request_handle_.nolockNonLocalCheck(key)) {
        return false;
      }

      return server.request_handle_.attemptLocalPull(key, vals->data(), false, false);
    }

    /**
     * \brief Make parameters local to this parameter server
     *
     *        Optionally, you can pass a pointer to an integer in which the method
     *        will store the number of already local keys
     *
     * @param keys the list of keys to localize
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Localize(const std::vector<Key>& keys, size_t* extNumLocal = nullptr) {

      // no need to relocate if there is only one node
      if (Postoffice::Get()->num_servers() == 1) return -1;

      int ts = obj_->NewRequest(kServerGroup, keys.size());

      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers()); // TODO: reuse

      size_t numLocal = 0;
      for (size_t i = 0; i < keys.size(); ++i) {
        Key key = keys[i];
        CHECK(key < Postoffice::Get()->max_key())<< "[ERROR] Localize key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->max_key() << ")";
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

      if(extNumLocal != nullptr) *extNumLocal = numLocal;

      // send out messages and return timestamp
      SendRequests(ts, false, requests, numLocal, Control::LOCALIZE);
      return ts;
    }

    /**
     * \brief Prepare a sample of K random keys
     *        Returns a sample id
     */
    SampleID PrepareSample(size_t K) {
      return server.sampling_->prepare_sample(K, customer_id_);
    }

    /**
     * \brief Pull N keys a previously prepared sample (N = `keys.size()`).
     *        Returns an operation timestamp as `Pull` does. (i.e., it is safe to use
     *        the values in `vals` only after you waited for the returned timestamp)
     */
    int PullSample(SampleID id, std::vector<Key>& keys, std::vector<Val>& vals) {
      return server.sampling_->pull_sample(id, keys, vals, customer_id_);
    }

    /**
     * \brief Declare a sample id as "finished" (such that the system can delete
     *        any related information)
     */
    void FinishSample(SampleID id) {
      server.sampling_->finish_sample(id, customer_id_);
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
     * \brief Wait until all updates that have reached the store by now
     *        are propagated to all replicas
     */
    void WaitReplicaSync() {
      server.replica_manager_.WaitReplicaSync();
    }

    /**
     * \brief Split a large push into smaller groups, pushing each group individually
     *        (to work around RAM limitations)
     */
    int StaggeredPush(const std::vector<Key>& all_keys, std::vector<Val>& all_vals,
                      const size_t group_size=100000) {
      std::vector<Key> keys {};
      std::vector<Val> vals {};
      keys.reserve(group_size);
      vals.reserve(group_size * all_vals.size() / all_keys.size());

      size_t pos = 0;
      for (size_t k = 0; k!=all_keys.size(); ++k) {
        Key key = all_keys[k];
        auto len = server.request_handle_.get_len(key);
        keys.push_back(key);

        std::copy_n(all_vals.begin() + pos, len, std::back_inserter(vals));
        pos += len;

        // push a group if (1) we have enough keys or (2) this is the last loop iteration
        if (keys.size() == group_size || k == all_keys.size()-1) {
          Push(keys, vals);
          keys.resize(0);
          vals.resize(0);
        }
      }
      return -1; // waited for the pushes above
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
     * \brief Return the length of the value of a key
     */
    size_t GetLen(Key key) const {
      return server.request_handle_.get_len(key);
    }

    /**
     * \brief Finalize this worker, i.e., wait for all requests and callbacks and run a barrier
     */
    void Finalize() {
      // wait for all requests to finish
      WaitAll();

      // wait for all callbacks to run
      KVWorker<Val>::WaitCallbacks();

      Barrier();
    }

    /**
     * \brief Begin the setup period, i.e., a period in which model parameters are initialized
     *        Push and pull operation within this time do not count into statistics
     *        Pushes are applied to replicas without an aggregation factor
     */
    void BeginSetup() {
      WaitReplicaSync();
      Barrier();
      if (customer_id_ == 1) {
        server.replica_manager_.setInit(true);
      }
    }

    /**
     * \brief End the setup period
     */
    void EndSetup() {
      WaitReplicaSync();
      Barrier();
      if (customer_id_ == 1) {
        server.replica_manager_.setInit(false);
      }
      ResetStats();
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
      num_pull_params_in_transfer = 0;
      num_push_params = 0;
      num_push_params_local = 0;
      num_push_params_in_transfer = 0;
      server.request_handle_.reset_replica_stats();
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
        if (!isLocalize(cmd)) { // add values (unless this request belongs to a localize, which contains only keys)
          msg.AddData(kvs.vals);
        }
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
    long num_pull_params_in_transfer=0, num_push_params_in_transfer=0;
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
        size_t len = 0, orig_pos = 0, recv_pos = 0, key_pos = 0;
        auto &requested_keys = *orig.first;
        auto senderRank = Postoffice::Get()->IDtoRank(msg.meta.sender);
        for (size_t i = 0; i != requested_keys.size(); ++i) {
          Key key = kvs.keys[key_pos];
          len = server.request_handle_.get_len(requested_keys[i]);
          if (requested_keys[i] == key) {
            // write vals to correct position
            std::copy_n(kvs.vals.begin() + recv_pos, len, (*orig.second).begin() + orig_pos);

            // update location caches
            if (Postoffice::Get()->use_location_caches()) {
              server.addressbook.updateCache(key, senderRank);
            }

            // move pointers forward
            ++key_pos;
            recv_pos += len;

            // early stop: stop when all keys of this response are processed
            if (key_pos == kvs.keys.size()) {
              break;
            }
          }
          // move pointer on value-array of original request forward
          orig_pos += len;
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
