/**
 *  Copyright (c) 2018 by Contributors
 */
#ifndef PS_COLOC_KV_WORKER_H_
#define PS_COLOC_KV_WORKER_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <tuple>
#include <queue>
#include "ps/kv_app.h"
#include "ps/addressbook.h"
#include "utils.h"
#include <boost/math/special_functions/factorials.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/dynamic_bitset.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace ps {

  // forward declarations
  template <typename Val, typename Handle>
    class ColoKVServer;
  template <typename Val, typename Handle>
    class SyncManager;

  // Holds information about one future intent of this worker. We use this
  // to store future intents separately from currently active intents.
  struct FutureIntent {
    FutureIntent(const Clock s, const Clock e,
                 std::shared_ptr<std::unordered_set<Key>>& _keys):
      start{s}, end{e}, keys{_keys} {}

    FutureIntent() {}

    Clock start;
    Clock end;
    std::shared_ptr<std::unordered_set<Key>> keys;
  };

  // we store future intents in a heap, with the "nearest" intents on top
  class CompareFutureIntent {
  public:
    bool operator() (const FutureIntent& lhs, const FutureIntent& rhs) const {
      return (lhs.start>rhs.start);
    }
  };

  // helper class that runs the postoffice setup (before anything else is done)
  // the Start() call waits until all nodes are ready
  class PostofficeSetup {
  public:
    PostofficeSetup(const int customer_id, const unsigned int start_total=1) {
      // start primary (worker or server) channel
      Postoffice::Get()->Start(customer_id, nullptr, true);

      // start additional channels(without barrier). used for additional PS threads
      for (unsigned int i=1; i!=start_total; ++i) {
        Postoffice::Get()->Start(customer_id+i, nullptr, false);
      }
    }
  };

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
     * \param customer_id the customer id which is unique locally
     */
    explicit ColoKVWorker(const int customer_id, ColoKVServer<Val, Handle>& sserver) :
      postoffice_setup {customer_id}, server(sserver), customer_id_(customer_id),
      new_intents (Postoffice::Get()->num_channels()),
      future_intents (Postoffice::Get()->num_channels()) {

      // create customer
      using namespace std::placeholders;
      this->obj_ = new Customer(0, customer_id, std::bind(&ColoKVWorker<Val, Handle>::Process, this, _1));
      // note: obj_ is deleted by the destructor of KVWorker

      server.registerWorker(customer_id_, this);
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
     * @param vals a pointer to the values that should be pushed
     * @param set a flag whether the value should be pushed (added) or set
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Push(Key* keys, size_t num_keys, Val* vals, const bool set=false) {
      ++num_push_ops;
      num_push_params += num_keys;

      size_t len = 0, pos = 0;

      // STEP 1: fast special case (if all parameters are local)
      destinations.resize(num_keys);
      size_t numLocal = 0;
      for (size_t i = 0; i < num_keys; ++i) {
        // try to process the request locally
        Key key = keys[i];
        CHECK(key < Postoffice::Get()->num_keys())<< "[ERROR] Push key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->num_keys() << ")";
        len = server.request_handle_.get_len(key);
        destinations[i] = server.localPushOrDirections(key, vals + pos, set);
        pos += len;
        numLocal += (destinations[i] == LOCAL);
      }
      num_push_params_local += numLocal;
      if (numLocal == num_keys) {
        ++num_push_ops_local;
        return -1; // fast case done
      }


      // STEP 2: slow general case (send out requests to other servers)
      int ts = obj_->NewRequest(kServerGroup, num_keys);

      auto nc = Postoffice::Get()->num_channels();
      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers() * nc);
      std::vector<size_t> requests_num_keys (requests.size());
      std::vector<size_t> requests_len_vals (requests.size());

      // determine how large the requests will be
      for (size_t i = 0; i!=num_keys; ++i) {
        if (destinations[i] != LOCAL) {
          Key key = keys[i];
          len = server.request_handle_.get_len(key);
          auto c = server.request_handle_.get_channel(key);
          requests_num_keys[destinations[i]*nc+c] += 1;
          requests_len_vals[destinations[i]*nc+c] += len;
        }
      }

      // allocate memory for keys and values in the requests
      for (size_t z = 0; z!=requests.size(); ++z) {
        requests[z].keys.reserve(requests_num_keys[z]);
        requests[z].vals.reserve(requests_len_vals[z]);
      }

      // copy keys and values to requests
      pos = 0; // reset for copy pass
      for (size_t i = 0; i!=num_keys; ++i) {
        Key key = keys[i];
        len = server.request_handle_.get_len(key);
        auto c = server.request_handle_.get_channel(key);
        if (destinations[i] != LOCAL) {
          requests[destinations[i]*nc+c].keys.push_back(key);
          std::copy_n(vals+pos, len, std::back_inserter(requests[destinations[i]*nc+c].vals));
        }
        pos += len;
      }

      // send out requests
      SendRequests(ts, true, requests, numLocal, 0, set);
      return ts;
    }

    /**
     * \brief Push key--values into the PS (overrides KVWorker::Push())
     *
     * This is a wrapper that takes a vector for keys and values
     *
     * @param keys a list of keys
     * @param vals the according values
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Push(std::vector<Key>& keys, std::vector<Val>& vals) {
      return Push(keys.data(), keys.size(), vals.data());
    }

    /**
     * \brief Push key--values into the PS (overrides KVWorker::Push())
     *
     * This is a wrapper that takes a vector of keys and a raw pointer for values
     *
     * @param keys a list of keys
     * @param vals the according values
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Push(std::vector<Key>& keys, Val* val_ptr) {
      return Push(keys.data(), keys.size(), val_ptr);
    }

    /**
     * \brief Set key--values in the PS
     *
     * Sets the keys' parameters to specific values
     *
     * @param keys a list of keys
     * @param vals the according values
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Set(Key* keys, size_t num_keys, Val* vals) {
      return Push(keys, num_keys, vals, true);
    }


    /**
     * \brief Set key--values in the PS
     *
     * Sets the keys' parameters to specific values
     *
     * @param keys a list of keys
     * @param vals the according values
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Set(std::vector<Key>& keys, std::vector<Val>& vals) {
      return Push(keys.data(), keys.size(), vals.data(), true);
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
     * @param vals an array to store the vals into
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Pull(Key* keys, size_t num_keys, Val* vals, const Callback& cb = nullptr) {
      ++num_pull_ops;
      num_pull_params += num_keys;

      size_t len = 0, pos = 0;

      // STEP 1: fast special case (if all parameters are local)
      size_t numLocal = 0;
      destinations.resize(num_keys);
      for (size_t i = 0; i < num_keys; ++i) {
        // try to process the request locally
        Key key = keys[i];

        CHECK(key < Postoffice::Get()->num_keys())<< "[ERROR] Pull key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->num_keys() << ")";

        len = server.request_handle_.get_len(key);
        destinations[i] = server.localPullOrDirections(key, vals+pos);

        pos += len;
        numLocal += (destinations[i] == LOCAL);
      }
      num_pull_params_local += numLocal;
      if (numLocal == num_keys) {
        ++num_pull_ops_local;
        if (cb) cb();
        return -1;
      }

      // Step 2: slow general case

      // remember which key positions were processed locally already
      auto processed = std::make_unique<boost::dynamic_bitset<>>(num_keys);
      for (size_t i=0; i!=num_keys; ++i) {
        (*processed)[i] = (destinations[i] == LOCAL);
      }

      // create request
      int ts = obj_->NewRequest(kServerGroup, num_keys);
      orig_kv_mu_.lock();
      request_original_kvs_[ts] = std::make_tuple(keys, num_keys, vals, std::move(processed));
      orig_kv_mu_.unlock();

      this->AddCallback(ts, [this, ts, keys, vals, cb]() mutable {
          orig_kv_mu_.lock();
          request_original_kvs_.erase(ts);
          orig_kv_mu_.unlock();
          if (cb) cb();
        });

      auto nc = Postoffice::Get()->num_channels();
      std::vector<KVPairs<Val>> requests (Postoffice::Get()->num_servers() * nc);

      // for all non-local keys, we send out a request
      for (size_t i = 0; i!=num_keys; ++i) {
        Key key = keys[i];
        if (destinations[i] != LOCAL) {
          auto c = server.request_handle_.get_channel(key);
          requests[destinations[i]*nc+c].keys.push_back(key);
        }
      }

      // send out messages and return timestamp
      SendRequests(ts, false, requests, numLocal, 0, false);

      return ts;
    }

    /**
     * \brief Pull key--values from the PS (overwrites KVWorker::Pull())
     *
     * This is a wrapper that takes a vector of keys and a (pointer to a) vector for values
     *
     * @param keys a list of keys
     * @param vals a vector to store the vals into
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Pull(std::vector<Key>& keys, std::vector<Val>* vals,
             const Callback& cb = nullptr) {
      return Pull(keys.data(), keys.size(), vals->data(), cb);
    }

    /**
     * \brief Pull key--values from the PS (overwrites KVWorker::Pull())
     *
     * This is a wrapper that takes a vector of keys and a raw pointer for values
     *
     * @param keys a list of keys
     * @param vals a vector to store the vals into
     * @return -1 if all the keys were answered locally, a timestamp otherwise
     */
    int Pull(std::vector<Key>& keys, Val* val_ptr) {
      return Pull(keys.data(), keys.size(), val_ptr);
    }

    /**
     * \brief Pull the value of a parameter if the parameter is local
     *        Returns true and stores value in *val if parameter is local, returns false otherwise
     */
    // raw pointer version
    bool PullIfLocal(const Key key, Val* vals) {
      CHECK(key < Postoffice::Get()->num_keys())<< "[ERROR] Pull key " << key << ", which is outside the configured key range [0,"<< Postoffice::Get()->num_keys() << ")";

      // try to check locality without acquiring a lock
      if (server.request_handle_.isNonLocal_noLock(key)) {
        return false;
      }

      return server.request_handle_.attemptLocalPull(key, vals, false);
    }
    // vector version
    bool PullIfLocal(const Key key, std::vector<Val>* vals) {
      return PullIfLocal(key, vals->data());
    }

    /**
     * \brief Signal intent for specific parameters, i.e., indicate that this
     *        worker will access the given parameters during a time period in the future.
     *
     *        This period is specified by the given `start` and `end` clocks.
     *        The `start` clock is inclusive, `end` clock is exclusive, i.e.,
     *        intent start=14 end=16 means that there is intent in clock 14 and 15
     *
     *        The `end` clock is optional. If the `end` clock is omitted, we
     *        assume that the intent is valid only for the `start` clock, i.e.,
     *        that the intent ends at clock `start`+1
    */
    // set of keys (move)
    int Intent(const std::unordered_set<Key>&& keys, Clock start, Clock end=0) {
      return ProcessIntent(std::move(keys), start, end);
    }

    // set of keys (copy)
    int IntentWithCopy(const std::unordered_set<Key>& keys, Clock start, Clock end=0) {
      std::unordered_set<Key> keys_copy (keys);
      return ProcessIntent(std::move(keys_copy), start, end);
    }

    // vector of keys
    int Intent(const std::vector<Key>& keys, Clock start, Clock end=0) {
      // make sure there are no duplicates
      std::unordered_set<Key> keys_set (keys.begin(), keys.end());
      return ProcessIntent(std::move(keys_set), start, end);
    }

    // array of keys
    int Intent(const Key* keys, const size_t num_keys, Clock start, Clock end=0) {
      // make sure there are no duplicates
      std::unordered_set<Key> keys_set (keys, keys+num_keys);
      return ProcessIntent(std::move(keys_set), start, end);
    }

    // single key
    int Intent(const Key key, Clock start, Clock end=0) {
      std::unordered_set<Key> keys_set {key};
      return ProcessIntent(std::move(keys_set), start, end);
    }


    /**
     * \brief Signal sampling intent

     I.e., indicate that this worker will access a parameter sample during the
     given time period. See `Intent()` for details on the period specification.

    */
    SampleID PrepareSample(size_t K, Clock start, Clock end=0) {
      return server.sampling_->prepare_sample(K, customer_id_, start, end);
    }

    /**
     * \brief Pull N keys a previously prepared sample (N = `keys.size()`).
     *        Returns an operation timestamp as `Pull` does. (i.e., it is safe to use
     *        the values in `vals` only after you waited for the returned timestamp)
     */
    // vector version
    int PullSample(SampleID id, std::vector<Key>& keys, std::vector<Val>& vals) {
      return server.sampling_->pull_sample(id, keys.data(), keys.size(), vals.data(), customer_id_);
    }
    // raw pointer version
    int PullSample(SampleID id, Key* keys, size_t num_keys, Val* vals) {
      return server.sampling_->pull_sample(id, keys, num_keys, vals, customer_id_);
    }

    /**
     * \brief Declare a sample id as "finished" (such that the system can delete
     *        any related information)
     */
    void FinishSample(SampleID id) {
      server.sampling_->finish_sample(id, customer_id_);
    }

    // helper: calculate an ID for this worker
    int workerId() const {
      return Postoffice::Get()->my_rank()*Postoffice::Get()->num_worker_threads()+customer_id_;
    }

    /**
     * \brief Advance this worker's clock
     */
    void advanceClock() {
      ++clock;
      assert(clock >= 0); // prevent overflows
    }

    /**
     * \brief Obtain the worker's current clock
     */
    Clock currentClock() const {
      return clock;
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
     * \brief Wait for 4 timestamps
     */
    void Wait(int ts1, int ts2, int ts3, int ts4) {
      if(ts1 != -1) obj_->WaitRequest(ts1);
      if(ts2 != -1) obj_->WaitRequest(ts2);
      if(ts3 != -1) obj_->WaitRequest(ts3);
      if(ts4 != -1) obj_->WaitRequest(ts4);
    }

    /**
     * \brief Wait for 5 timestamps
     */
    void Wait(int ts1, int ts2, int ts3, int ts4, int ts5) {
      if(ts1 != -1) obj_->WaitRequest(ts1);
      if(ts2 != -1) obj_->WaitRequest(ts2);
      if(ts3 != -1) obj_->WaitRequest(ts3);
      if(ts4 != -1) obj_->WaitRequest(ts4);
      if(ts5 != -1) obj_->WaitRequest(ts5);
    }

    /**
     * \brief Wait for one synchronization round to start (in each channel).
     *        I.e., wait that all updates are sent out to their corresponding
     *        main copies.
     */
    void WaitSync() {
      // no need to wait if there is only 1 node
      if (Postoffice::Get()->num_servers() == 1) {
        return;
      }

      auto num_channels = Postoffice::Get()->num_channels();

      // determine current sync round (for each channel)
      std::vector<unsigned long> wait_syncs (num_channels);
      for (unsigned int c=0; c!=num_channels; ++c) {
        wait_syncs[c] = server.sync_managers_[c].GetCurrentSync() + 2;
      }

      // wait for all sync managers to finish 2 sync rounds
      while (true) {
        bool allFinished = true;
        for (unsigned int c=0; c!=num_channels; ++c) {
          if (server.sync_managers_[c].GetCurrentSync() < wait_syncs[c]) {
            allFinished = false;
            break;
          }
        }

        if (allFinished) {
          break;
        }

        // wait briefly until we try again
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }

      ALOG("r" << Postoffice::Get()->my_rank() << "-w" << workerId() << ": Waited for syncs " << str(wait_syncs));
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
      server.deregisterWorker(customer_id_);

      // wait for all requests to finish
      WaitAll();

      // wait for all callbacks to run
      KVWorker<Val>::WaitCallbacks();

      // wait for one sync
      WaitSync();

      Barrier();

      Postoffice::Get()->Finalize(customer_id_, false);

      // propagate stats to server
      server.num_pull_ops += num_pull_ops;
      server.num_pull_ops_local += num_pull_ops_local;
      server.num_push_ops += num_push_ops;
      server.num_push_ops_local += num_push_ops_local;
      server.num_pull_params += num_pull_params;
      server.num_pull_params_local += num_pull_params_local;
      server.num_push_params += num_push_params;
      server.num_push_params_local += num_push_params_local;
    }

    /**
     * \brief Begin the setup period, i.e., a period in which model parameters are initialized
     *        Push and pull operation within this time do not count into statistics
     *        Pushes are applied to replicas without an aggregation factor
     */
    void BeginSetup() {
      WaitSync();
      Barrier();
    }

    /**
     * \brief End the setup period
     */
    void EndSetup() {
      WaitSync();
      WaitSync();
      Barrier();
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
      num_push_params = 0;
      num_push_params_local = 0;
      server.request_handle_.reset_replica_stats();
    }

    /**
     * \brief Wait for a barrier among all servers
     */
    void Barrier() {
      Postoffice::Get()->Barrier(customer_id_, kWorkerThreadGroup);
    }

    size_t GetNumKeys() const {
      return Postoffice::Get()->num_keys();
    }

    // get future intents of this worker until a given clock, and move them to
    // the passed map
    void getNewRelevantIntents(const unsigned int channel, const Clock clock_current, const Clock clock_until,
                               std::unordered_map<Key, std::vector<WorkerClockPair>>& relevant_intents) {
      // add the intents that were newly added by the worker since the last
      // synchronization round to the heap of all future intents
      FutureIntent intent {};
      while (new_intents[channel].pop(intent)) {
        future_intents[channel].push(std::move(intent));
      }

      // extract intents that are relevant for this synchronization round
      while (!future_intents[channel].empty() &&
             future_intents[channel].top().start <= clock_until) {

        // move this intent to the by-key map
        auto& top = future_intents[channel].top();
        for (Key key : *top.keys.get()) {
          if (server.request_handle_.get_channel(key) == channel) {
            if (clock_current < top.end) { // the intent is useful only if the corresp. worker has not passed the end clock already
              relevant_intents[key].emplace_back(customer_id_, top.end);
            }
          }
        }
        future_intents[channel].pop();
      }
    }

  private:

    /**
     * \brief Process intent signal
     *
     *        Assumes that `keys` contains no duplicates.
     *
     *        The `start` clock is inclusive, `end` clock is exclusive, i.e.,
     *        intent start=14 end=16 means that there is intent in clock 14 and 15
     *
     *        If the `end` clock is omitted, we assume that the intent is valid only
     *        for the `start` clock, i.e., that the intent ends at clock `start`+1
     */
    int ProcessIntent(const std::unordered_set<Key>&& keys, Clock start, Clock end=0) {
      // end clock can be omitted
      if (end == 0) end = start+1;

      // no need to send intents etc if there is only one node
      if (Postoffice::Get()->num_servers() == 1) return -1;

      // We store the full set multiple times: once per channel. The
      // synchronization manager of each channel will parse the set later on,
      // picking out the keys for its channel. The smart pointer makes sure that
      // the set is deallocated when all channels have parsed the set.
      auto keys_shared = std::make_shared<std::unordered_set<Key>>(std::move(keys));

      FutureIntent intent {start, end, keys_shared};

      for (unsigned int channel=0; channel!=Postoffice::Get()->num_channels(); ++channel) {
        while(!new_intents[channel].push(intent)) {
          ALOG("[WARNING] The queue for incoming intent of worker " << workerId() << " for channel " << channel << " is full. This hinders performance. You should increase the size of the incoming queues. To do so, increase SIZE in `boost::lockfree::capacity<SIZE>` in coloc_kv_worker.h.");
        }
      }
      return -1;
    }

    /**
     * \brief Send requests to other parameter servers to answer push/pull/localize calls
     */
    void SendRequests(const int timestamp, const bool push,
                      const std::vector<KVPairs<Val>>& requests,
                      const int numLocal, int cmd=0, const bool set=false) {
      if (timestamp != WITHOUT_TS && numLocal != 0) { // add response count for the requests that were answered already
        obj_->AddResponse(timestamp, numLocal);
      }
      // untypical case: have all responses already -> run callback right now
      if (timestamp != WITHOUT_TS && obj_->HasAllResponses(timestamp)) {
        this->RunCallback(timestamp);
        obj_->NotifyThreads();
      }

      // send out messages to other servers
      auto ns = Postoffice::Get()->num_servers();
      auto nc = Postoffice::Get()->num_channels();
      assert(requests.size() == nc * ns);
      for (int r=0; r!=ns; ++r) {
        for (unsigned int c=0; c!=nc; ++c) {
          if (!requests[r*nc+c].keys.empty()) {
            this->Request(r, requests[r*nc+c], push, cmd, timestamp, set, c);
          }
        }
      }
    }

    /**
     * \brief Send out one request to a specific parameter server
     */
    void Request(const int destination, const KVPairs<Val>& kvs, const bool push,
                 const int cmd, const int timestamp, const bool set, const unsigned int channel) {
      // Message object
      Message msg;
      msg.meta.app_id = obj_->app_id();
      msg.meta.customer_id = obj_->customer_id();
      msg.meta.request     = true;
      msg.meta.push        = push;
      msg.meta.set         = set;
      msg.meta.channel     = channel;
      msg.meta.head        = cmd;
      msg.meta.timestamp   = timestamp;
      msg.meta.recver      = Postoffice::Get()->ServerRankToID(destination);
      if (kvs.keys.size()) {
        msg.AddData(kvs.keys);
        msg.AddData(kvs.vals);
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

    // helper object to set up postoffice before anything else happens
    PostofficeSetup postoffice_setup;

    /** \brief reference to the parameter server of this process */
    ColoKVServer<Val, Handle>& server;

    // pointers to original keys and values (we use this for pushs: when a message arrives, we put the arriving values directly into the original array)
    std::unordered_map<int, std::tuple<const Key*, size_t, Val*, std::unique_ptr<boost::dynamic_bitset<>>>> request_original_kvs_;
    std::mutex orig_kv_mu_;

    // The customer id of the thread that owns this KVWorker instance (used for barrier)
    const int customer_id_;

    // the time of one of the first clocks (we use this to estimate clock rate)
    std::chrono::time_point<std::chrono::high_resolution_clock> start_clock;

    // The status (either local or a destination) of each key of an operation
    // The vector is reused across operations in this worker thread
    std::vector<int> destinations {};

    // This worker's clock
    std::atomic<Clock> clock {0};

    // All intents that this worker has signaled since the last synchronization round
    std::vector<boost::lockfree::spsc_queue<FutureIntent, boost::lockfree::capacity<65536>>> new_intents;

    // This worker's future intents (one heap per channel)
    std::vector<std::priority_queue<FutureIntent, std::vector<FutureIntent>, CompareFutureIntent>> future_intents;

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

        // get reference to original keys + val arrays of this timestamp
        orig_kv_mu_.lock();
        auto& orig = request_original_kvs_[ts];
        orig_kv_mu_.unlock();

        auto requested_keys = std::get<0>(orig);
        auto num_requested_keys = std::get<1>(orig);
        auto requested_vals = std::get<2>(orig);
        auto processed = *(std::get<3>(orig));

        CHECK(requested_keys) << "FATAL: have no place to put reply to a pull requests at worker " << Postoffice::Get()->my_rank() << "::" << obj_->customer_id() << " for ts " << ts <<". Pointer is " << requested_keys;

        // go trough requested keys and match with the keys of this message,
        // place received vals at the correct positions of the array
        // note: we can do this without lock because (a) there is only one receive thread
        //    (b) if there were multiple, there is separated write areas in the val array
        size_t len = 0, orig_pos = 0, recv_pos = 0, key_pos = 0;
        auto senderRank = Postoffice::Get()->IDtoRank(msg.meta.sender);

        for (size_t i = 0; i != num_requested_keys; ++i) {
          Key key = kvs.keys[key_pos];
          len = server.request_handle_.get_len(requested_keys[i]);
          if (requested_keys[i] == key && !processed[i]) {
            // write vals to correct position
            std::copy_n(kvs.vals.begin()+recv_pos, len, requested_vals+orig_pos);
            processed[i] = 1; // mark as processed

            // update location caches
            if (Postoffice::Get()->use_location_caches()) {
              server.addressbook.updateLocationCache(key, senderRank);
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

        // did we run through all keys?
        assert(key_pos == kvs.keys.size());

        // increase receive count
        obj_->AddResponse(ts, kvs.keys.size());
      } else if(msg.meta.push && msg.data.size()) { // push call
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
