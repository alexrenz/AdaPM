/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_COLOC_KV_SERVER_H_
#define PS_COLOC_KV_SERVER_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <valarray>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <algorithm>
#include <atomic>
#include "ps/kv_app.h"
#include "ps/addressbook.h"
#include <limits>
#include "ps/replica_manager.h"
#include "ps/sampling.h"

#include <iostream>
#include <sstream>

namespace ps {

#ifdef PS_RELOC_STATS
  long STATS_total_transfer_time_sum = 0;
  long STATS_total_transfers = 0;
  long STATS_total_transfer_time_max = 0;
  long STATS_total_transfer_time_min = std::numeric_limits<long>::max();
  std::atomic<long> STATS_num_localizes (0);
  std::atomic<long> STATS_num_localizes_local (0);
  std::atomic<long> STATS_num_localizes_queue (0);
#endif

  /**
   * \brief The information necessary to notify a waiting worker thread
   */
  struct WaitingThread {
    WaitingThread() {
      CHECK(false) << "Default constructor of WaitingThread should not be called";
    }
  WaitingThread(Customer* c, int timestamp): customer{c}, ts{timestamp} {}
    Customer* customer;
    int ts;
  };

  /**
   * \brief A message that is queued until one or more keys that will be local soon
   *        at this server will be local
   */
  template<typename Val>
  struct QueuedMessage {
    KVPairs<Val> kvs;
    KVMeta meta;
    bool sendOnDelete = false;
  };

  /**
   * \brief Queue of requests for a key while this key is being localized to this server
   */
  // TODO: move this to transfers class
  template <typename Val>
    struct Transfer {
      bool _ongoing = false;
      std::vector<std::pair<bool,Val*>> ops; // queued pull and push operations: <true=push/false=pull, pointer_to_val_location>
      std::vector<std::shared_ptr<KVPairs<Val>>> push_data_ptrs; // to ensure that push data is still around when transfer is finished
    std::vector<std::vector<Val>> pushs; // holds copies of to-push values (when they are necessary)
      std::vector<WaitingThread> threads; // waiting local worker threads
      std::vector<std::shared_ptr<QueuedMessage<Val>>> messages; // waiting messages for remote worker threads
      bool subsequentLocalize = false; // did we receive a subsequent localize? (if yes, we forward the key directly when we receive it)
      std::shared_ptr<QueuedMessage<Val>> subsequentLocalizeMsg; // did we receive a subsequent localize? (if yes, we forward the key directly when we receive it)
#ifdef PS_RELOC_STATS
      util::Stopwatch total_transfer_time;
#endif

      Transfer() {}

      Transfer(Transfer&& q) { // move constructor
        ops.swap(q.ops);
        pushs.swap(q.pushs);
        push_data_ptrs.swap(q.push_data_ptrs);
        threads.swap(q.threads);
        messages.swap(q.messages);
        std::swap(subsequentLocalize, q.subsequentLocalize);
        std::swap(subsequentLocalizeMsg, q.subsequentLocalizeMsg);
        std::swap(_ongoing, q._ongoing);
#ifdef PS_RELOC_STATS
        std::swap(total_transfer_time, q.total_transfer_time);
#endif
      }

      // Copy constructor
      Transfer(const Transfer& q) {
        CHECK(false) << "Copy constructor for transfer should not be called";
      }

      // note a thread as waiting. When the transfer finishes, the thread will receive a wake signal
      void addThread (Customer* customer, int ts) {
        CHECK(_ongoing) << "Add thread to queue, but transfer is not ongoing";
        threads.push_back(WaitingThread(customer, ts));
      }

      // start a transfer
      void start() {
        _ongoing = true;
#ifdef PS_RELOC_STATS
        total_transfer_time.start();
#endif
      }

      // stop a transfer, i.e., reset the transfer object
      void reset() {
        CHECK(_ongoing) << "Resetting transfer that is not ongoing";
        _ongoing = false;
        ops.resize(0);
        pushs.resize(0);
        push_data_ptrs.resize(0);
        threads.resize(0);
        messages.resize(0);
        subsequentLocalize = false;
        subsequentLocalizeMsg.reset();

        // statistics
#ifdef PS_RELOC_STATS
        total_transfer_time.stop();
        STATS_total_transfer_time_sum += total_transfer_time.elapsed_ns();
        ++STATS_total_transfers;
        STATS_total_transfer_time_max = max(total_transfer_time.elapsed_ns(), STATS_total_transfer_time_max);
        STATS_total_transfer_time_min = min(total_transfer_time.elapsed_ns(), STATS_total_transfer_time_min);
#endif
      }

    public:
      inline bool ongoing() { return _ongoing; }
    };


  /**
   * \brief A server node for maintaining key-value pairs
   */
  template <typename Val, typename Handle>
    class ColoKVServer : public KVServer<Val> {
    friend ColoKVWorker<Val, Handle>;
  public:
    /**
     * \brief constructor
     *        The system will replicate all keys that are contained in `keys_to_replicate`.
     *        Replication will be disabled entirely (i.e., no replica mgr thread) if
     *        `keys_to_replicate` is a nullptr or contains 0 keys.
     */
    explicit ColoKVServer(int app_id, Handle& h, vector<Key>* keys_to_replicate=nullptr) :
      KVServer<Val>(app_id), request_handle_(h), my_rank{Postoffice::Get()->my_rank()},
      replica_manager_{h, keys_to_replicate} {

      // start the replica manager thread
      if (keys_to_replicate != nullptr && keys_to_replicate->size() != 0) {
        replica_manager_thread_  = std::thread (&ReplicaManager<Val,Handle>::thread, &replica_manager_);
        SET_THREAD_NAME((&replica_manager_thread_), "replica-manager");
      }
      lifetime.start();
    }

    ~ColoKVServer() {
      delete sampling_;
    }

    /**
     * \brief Forward a pull/push request to another parameter server
     */
    void Forward(int destination, const KVMeta& req, const KVPairs<Val>& fw);

    /**
     * \brief Shut down server (mostly writing statistics)
     */
    void shutdown() {
      lifetime.stop();
      // stop the replica manager thread
      if (replica_manager_thread_.joinable()) {
        replica_manager_.stop();
        replica_manager_thread_.join();
      }

      // terminate any sampling support activities
      if (sampling_ != nullptr) {
        sampling_->terminate();
      }

      // output per-server locality statistics
      ADLOG(//pulls
            "server " << my_rank << ": " << num_pull_params << " parameters pulled, " <<
            100.0 * num_pull_params_local / num_pull_params << "% local (~" <<
            100.0 * request_handle_.get_num_pulls_to_replicas() / num_pull_params << "% to replicas), " <<
            100.0 * num_pull_params_in_transfer / num_pull_params << "% in transfer (" <<
            num_pull_ops << " ops, " << 100.0 * num_pull_ops_local / num_pull_ops << "% local)\n" <<
            // pushs
            "server " << my_rank << ": " << num_push_params << " parameters pushed, " <<
            100.0 * num_push_params_local / num_push_params << "% local (~" <<
            100.0 * request_handle_.get_num_pushs_to_replicas() / num_push_params << "% to replicas), " <<
            100.0 * num_push_params_in_transfer / num_push_params << "% in transfer (" <<
            num_push_ops << " ops, " << 100.0 * num_push_ops_local / num_push_ops << "% local)"
            );

      // output relocation statistics
#ifdef PS_RELOC_STATS
      if (STATS_total_transfers == 0) STATS_total_transfers = -1; // prevent division by 0
      auto alive = lifetime.elapsed_s();
      ADLOG("Relocations at s" << my_rank << ": " << STATS_total_transfers << " (" << STATS_num_localizes_local <<
            " already local, " << STATS_num_localizes_queue <<  " already running, " << STATS_num_localizes <<
            " localizes). Total transfer time: " << STATS_total_transfer_time_sum << " ns. " <<
            STATS_total_transfer_time_sum/STATS_total_transfers << " ns per transfer (" <<
            STATS_total_transfer_time_min << " min / " << STATS_total_transfer_time_max << " max). " <<
            STATS_total_transfers/alive << "r/s in " << alive << "s");
#endif

      // write handle access statistics
      request_handle_.writeStats();
    }

    /**
     * \brief Enable support for sampling
     */
    void enable_sampling_support(Key (*const sample_key)()) {
      auto& sst = SamplingSupport<Val, ColoKVWorker<Val,Handle>>::sampling_strategy;
      switch (sst) {
      case SamplingSupportType::Naive:
        sampling_ = new NaiveSamplingSupport<Val, ColoKVWorker<Val, Handle>>(sample_key);
        break;
      case SamplingSupportType::Preloc:
        sampling_ = new PrelocSamplingSupport<Val, ColoKVWorker<Val, Handle>>(sample_key);
        break;
      case SamplingSupportType::Pool:
        sampling_ = new PoolSamplingSupport<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>(sample_key, this);
        break;
      case SamplingSupportType::OnlyLocal:
        sampling_ = new OnlyLocalSamplingSupport<Val, ColoKVWorker<Val, Handle>>(sample_key);
        break;
      default:
        ALOG("Unkown sampling support type '" << sst << "'. Aborting.");
        abort();
        break;
      }
    }

    // execute a barrier among all servers
    void Barrier() {
      Postoffice::Get()->Barrier(0, kServerGroup) ;
    }

    // system options
    static unsigned int keepAroundUnused; // if >0: max. microseconds to keep around the value of an unused parameter

    // add system options to an application
    static void AddSystemOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sys.keep_unused", po::value<unsigned int>(&keepAroundUnused)->default_value(0), "system option: keep around unused parameters (0: disabled, >0: max time to keep around (in microseconds)")
        ("sys.nw_threads", po::value<int>(&Postoffice::Get()->num_network_threads_)->default_value(3), "number of network threads")
        ;

      // replication options
      ReplicaManager<Val,Handle>::AddReplicationOptions(options);

      // sampling options
      SamplingSupport<Val,ColoKVWorker<Val, Handle>>::AddSamplingOptions(options);
    }

  private:

    // Send out forward messages (to other servers)
    void sendForwards(std::vector<KVPairs<Val>>& forwards, KVMeta& meta) {

      for (unsigned int i=0; i!=forwards.size(); ++i) {
        if (!forwards[i].keys.empty()) {
          // resize the possibly over-allocated vals array (unless the message contains only keys)
          if (!forwards[i].vals.empty()) {
            forwards[i].vals.resize(request_handle_.get_total_len(forwards[i].keys));
          }

          this->Forward(i, meta, forwards[i]);
        }
      }
    }

    Handle& request_handle_; // hides the request_handle_ of KVServer
    bool Process(const Message& msg); // we re-define Process to use the correct request_handle_
    void ProcessPushPullRequest(std::shared_ptr<KVPairs<Val>>& data, KVPairs<Val>& res,
                                std::shared_ptr<QueuedMessage<Val>>& queued_msg);
    void ProcessLocalizeRequest(KVPairs<Val>& data, KVPairs<Val>& res,
                           std::shared_ptr<QueuedMessage<Val>>& queued_msg);
    void ProcessLocalizeResponse(const Message& msg);
    Addressbook addressbook {};
    int my_rank; // the rank of this server

    // replica management
    ReplicaManager<Val,Handle> replica_manager_;
    std::thread         replica_manager_thread_;

    // sampling management
    SamplingSupport<Val, ColoKVWorker<Val, Handle>>* sampling_=nullptr;

    util::Stopwatch lifetime; // lifetime of this server

    long num_pull_ops=0, num_pull_ops_local=0, num_push_ops=0, num_push_ops_local=0;
    long num_pull_params=0, num_pull_params_local=0, num_push_params=0, num_push_params_local=0;
    long num_pull_params_in_transfer=0, num_push_params_in_transfer=0;
    long num_pull_params_rep=0, num_push_params_rep=0;
  };

  // define (static) program options
  template <typename Val, typename Handle> unsigned int ColoKVServer<Val,Handle>::keepAroundUnused;


  /**
   * \brief Process a request to a parameter server
   */
  template <typename Val, typename Handle>
    bool ColoKVServer<Val, Handle>::Process(const Message& msg) {

    if (msg.meta.request) { // process requests (push, pull, and localize)

      // create a shared pointer to a (potentially delayed) response
      std::shared_ptr<QueuedMessage<Val>> queued_msg(new QueuedMessage<Val>(), [this] (QueuedMessage<Val>* qm) {
          // we send the message when all "local soon" keys have been answered
          if(qm->sendOnDelete) {
            this->Response(qm->meta, qm->kvs);
          }
          delete qm;
        });

      // prepare response header
      KVMeta& meta = queued_msg.get()->meta;
      meta.cmd       = msg.meta.head;
      meta.push      = msg.meta.push;
      meta.sender    = msg.meta.sender;
      meta.timestamp = msg.meta.timestamp;
      meta.customer_id = msg.meta.customer_id;

      // parse request key/value data
      std::shared_ptr<KVPairs<Val>> data_ptr (new KVPairs<Val>());
      data_ptr->keys = msg.data[0];
      if (msg.data.size() >= 2) data_ptr->vals = msg.data[1];
      KVPairs<Val>& data = *data_ptr;

      // prepare response key/value data
      KVPairs<Val>& res = queued_msg.get()->kvs;
      if (!meta.push) {
        // don't know exactly how many keys will be answered locally
        // overallocate (largest possible case: all keys are local)
        res.vals.resize(request_handle_.get_total_len(data.keys), 0, false);
      }

      // process requests
      if (isLocalize(meta.cmd)) {
        ProcessLocalizeRequest(data, res, queued_msg);
      } else {
        ProcessPushPullRequest(data_ptr, res, queued_msg);
      }
    } else if(isLocalize(msg.meta.head)) { // process responses to localizes
      // note: push/pull responses are handled in the worker threads (i.e., not here)
      ProcessLocalizeResponse(msg);
    } else {
      LL << "FATAL: unknown type of message in server thread";
    }
    return false;
  }

  /**
   * \brief Process a regular (i.e., push/pull) request to a parameter server
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::ProcessPushPullRequest(std::shared_ptr<KVPairs<Val>>& data_ptr, KVPairs<Val>& res,
                                                           std::shared_ptr<QueuedMessage<Val>>& queued_msg) {

    // prepare a couple of forward messages to other servers
    std::vector<KVPairs<Val>> forwards(Postoffice::Get()->num_servers());

    KVMeta& meta = queued_msg.get()->meta;

    KVPairs<Val>& data = *data_ptr;

    uint numLocal = 0;
    uint numLocalSoon = 0;
    size_t len = 0, data_pos = 0, res_pos = 0;
    // for each key, attempt a local pull(/push). forward the keys that are not local

    for (size_t i = 0; i < data.keys.size(); ++i) {
      Key key = data.keys[i];
      len = request_handle_.get_len(key);
      ps::Status status;
      if (meta.push) // push request
        status = request_handle_.processPush(key, data.vals.begin() + data_pos, -1, 0, false, queued_msg, data_ptr);
      else // pull request
        status = request_handle_.processPull(key, res.vals.data() + res_pos, -1, 0, false, queued_msg);

      if (status == ps::Status::LOCAL) {
        if (!meta.push) { // add key to reply for pull requests
          res.keys.push_back(key);
          res_pos += len;
        }
        ++numLocal;
      } else if (status == ps::Status::IN_TRANSFER) {
        res_pos += len;
        ++numLocalSoon;
      } else {
        auto destination = addressbook.getDirections(key); // either the manager or the owner (no caches)
        forwards[destination].keys.push_back(key);
        if (meta.push) { // forward the "to push" values to the responsible server
          std::copy_n(data.vals.begin() + data_pos, len, std::back_inserter(forwards[destination].vals));
        }
      }
      data_pos += len;
    }

    // if anything was or will be answered locally, send a response to the original sender
    // (either right now or later)
    if (numLocal != 0 || numLocalSoon != 0) {
      if(meta.push) {
        // for a push, we send back the number of locally processed keys
        res.vals.resize(0);
        res.keys.resize(1);
        res.keys[0] = numLocal + numLocalSoon;
      } else {
        res.vals.resize(request_handle_.get_total_len(res.keys));
        // need to do this also for the direct message
      }

      if (numLocalSoon == 0) {
        this->Response(meta, res);
      } else {
        // will send response later (when all pips are answered)
        queued_msg.get()->sendOnDelete = true;
      }
    }

    // if any key was not answered locally, send forward message(s)
    if (numLocal + numLocalSoon < data.keys.size()) {
      sendForwards(forwards, meta);
    }
  }

  /**
   * \brief Process a localize request to a parameter server
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::ProcessLocalizeRequest(KVPairs<Val>& data, KVPairs<Val>& res,
                                                      std::shared_ptr<QueuedMessage<Val>>& queued_msg) {
    KVMeta& meta = queued_msg.get()->meta;
    auto senderRank = Postoffice::Get()->IDtoRank(meta.sender);

    if (meta.cmd == Control::LOCALIZE) { // first message of a localize: Localize(param), requester->manager
      // prepare key/value data for TRANSFER messages (one to each involved old owner)
      std::vector<KVPairs<Val>> transfers(Postoffice::Get()->num_servers());
      std::vector<short> destinations(data.keys.size()); // store old owners

      // update residence for all keys in this localize request
      addressbook.lock();
      for (size_t i = 0; i < data.keys.size(); ++i) {
        Key key = data.keys[i];
        if(addressbook.isManagedHere(key)) {
          destinations[i] = addressbook.updateResidenceUnsafe(key, senderRank);
          TLOG(key, senderRank, "manager received LOCALIZE. update residence ");
        } else {
          // an exception: localize requester was also the manager, so it sent the message to the owner directly
          // thus, we don't need to update ownership
          TLOG(key, senderRank, "manager received LOCALIZE. not manager, don't update residence ");
        }
      }
      addressbook.unlock();

      // decide what to to with each parameter in this message. options
      // 1) this PS is current owner: take out of local PS and send TRANSFER response to requester
      // 2) this PS is the next designated owner: wait until transfer is finished, then send TRANSFER to requester
      // 3) send HAND_OVER message to current owner
      size_t len = 0, pos = 0;
      request_handle_.lockAll(); // does nothing if we use LOCK_SINGLE locking strategy
      for (size_t i = 0; i < data.keys.size(); ++i) {
        Key key = data.keys[i];
        len = request_handle_.get_len(key);
        request_handle_.lockSingle(key); // does nothing if we use LOCK_ALL locking strategy
        bool local = request_handle_.attemptLocalPullUnsafe(key, res.vals.data() + pos);
        if (local) {
          // take out of data structure
          res.keys.push_back(key);
          request_handle_.removeKeyUnsafe(key);
          TLOG(key, senderRank, "manager received LOCALIZE. also local. take out and send TRANSFER");
          pos += len;
        } else {

          if (destinations[i] == my_rank || !addressbook.isManagedHere(key)) { // HACK
            // two exceptions. in both cases, we send the TRANSFER message when the ongoing transfer is finished
            // 1) this node is the next designated owner and the node is currently in transfer
            // 2) the requester is also the manager and sent a LOCALIZE (instead of a HAND_OVER), but the parameter is still in transfer
            bool queued = request_handle_.transfers.noteSubsequentLocalizeUnsafe(key, queued_msg);
            TLOG(key, senderRank, "manager received LOCALIZE. in transfer. wait and send TRANSFER later");
            CHECK(queued) << "FATAL. got a LOCALIZE for key " << key << " and it should be in transfer, but it isn't. rank " << destinations[i] << " vs " << my_rank << ", " << (addressbook.isManagedHere(key) ? "managed here" : "not managed here");
          } else {
            transfers[destinations[i]].keys.push_back(key);
            TLOG(key, senderRank, "manager received LOCALIZE. send HAND_OVER to owner " << destinations[i]);
          }
        }
        request_handle_.unlockSingle(key);
      }
      request_handle_.unlockAll();

      // send out one HAND_OVER message to each involved old owner
      meta.cmd = Control::LOCALIZE_HAND_OVER;
      sendForwards(transfers, meta); // sends messages only to affected servers

      // if this node was the owner of any key, send a TRANSFER response to the requester
      if (res.keys.size() > 0) {
        meta.cmd = Control::LOCALIZE;
        res.vals.resize(request_handle_.get_total_len(res.keys));
        this->Response(meta, res);
      }
    } else if (meta.cmd == Control::LOCALIZE_HAND_OVER) { // second message of a transfer: HAND_OVER(param), manager->owner
      // for each parameter, check whether it is in the local PS
      // if yes, take it out of PS and send the TRANSFER message to the requester
      // if not, then it is in transfer to this node right now. wait for the transfer to finish, then send TRANSFER to requester
      size_t len = 0, pos = 0;
      request_handle_.lockAll();
      for (size_t i = 0; i < data.keys.size(); ++i) {
        Key key = data.keys[i];
        len = request_handle_.get_len(key);
        request_handle_.lockSingle(key);
        bool local = request_handle_.attemptLocalPullUnsafe(key, res.vals.data() + pos);

        if (local) {
          // take parameter out of local PS and transfer to requester
          res.keys.push_back(key);
          request_handle_.removeKeyUnsafe(key);
          TLOG(key, senderRank, "owner received HAND_OVER. take out and send TRANSFER");
          pos += len;
        } else {
          // must be in transfer. note down and send response later
          bool queued = request_handle_.transfers.noteSubsequentLocalizeUnsafe(key, queued_msg);
          TLOG(key, senderRank, "owner received HAND_OVER. key in transfer. queued TRANSFER, send when previous transfer is finished");
          CHECK(queued) << "FATAL. got a hand_over for key " << key << ", but key is not in transfer at rank " << my_rank;
        }
        request_handle_.unlockSingle(key);
      }
      request_handle_.unlockAll();

      // send TRANSFER response to localize requester
      if (res.keys.size() > 0) {
        meta.cmd = Control::LOCALIZE;
        res.vals.resize(request_handle_.get_total_len(res.keys));
        this->Response(meta, res);
      }

      // update location caches
      if (Postoffice::Get()->use_location_caches()) {
        for (size_t i = 0; i < data.keys.size(); ++i) {
          Key key = data.keys[i];
          addressbook.updateCacheUnsafe(key, senderRank);
        }
      }
    } else {
      ADLOG("Unkown type of localize request");
    }
  }

  /**
   * \brief Process a localize response, i.e., the last message of a localize.
   *        This includes storing the localized parameter in the local parameter server,
   *        processing queued requests and notifying waiting threads.
   *        If there is a subsequent localize for a parameter, we directly forward the parameter.
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::ProcessLocalizeResponse(const Message& msg) {
    KVPairs<Val> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];

    size_t len = 0, pos = 0;
    // Lists of threads to notify
    std::vector<std::vector<WaitingThread>> thread_lists (kvs.keys.size());
    // Parameters for which a subsequent localize has arrived, grouped by the next owner
    std::unordered_map<KVMeta*,std::pair<std::shared_ptr<QueuedMessage<Val>>,KVPairs<Val>>> subsequent_localize_map;

    request_handle_.lockAll(); // START CRITICAL SECTION.

    for (size_t i = 0; i < kvs.keys.size(); ++i) {
      Key key = kvs.keys[i];
      len = request_handle_.get_len(key);
      request_handle_.lockSingle(key);
      Val* val = &kvs.vals[pos];
      TLOG(key, my_rank, "requester received TRANSFER. localize finished");
      // manage finished localize for this key

      CHECK(request_handle_.transfers.isInTransferUnsafe(key)) << "FATAL! Did not find key " << key << " in pip map on rank " << my_rank;
      auto& queue = request_handle_.transfers.getTransferUnsafe(key);

      // process queued pull and push operations
      for (size_t r=0; r!=queue.ops.size(); ++r) {
        bool push = queue.ops[r].first; // is this op a push or a pull?
        Val* queued_val = queue.ops[r].second;
        if (push) { // process the queued push: merge into received value
          request_handle_.mergeValue(val, queued_val, len);
        } else { // process the queued pull: copy current value to stored destination
          std::copy_n(val, len, queued_val);
        }
      }

      if (queue.subsequentLocalize) {
        // the next (subsequent) localize for this parameter has already arrived. so we don't
        // put the parameter into the local PS. Instead, we forward to the next owner.

        // we group the subsequent localizes by the next owner and send one message
        // per (next_owner,timestamp) instead of one message per subsequent localizes

        // find the the message for this (next_owner,timestamp)
        KVMeta* meta = &(queue.subsequentLocalizeMsg->meta);
        auto search = subsequent_localize_map.find(meta);
        if(search == subsequent_localize_map.end()) {
          auto insert = subsequent_localize_map.insert({meta, {queue.subsequentLocalizeMsg, {}}});
          search = insert.first;
        }
        // add this parameter (key and value) to this message
        auto& subseq_kvs = search->second.second;
        subseq_kvs.keys.push_back(key);
        std::copy_n(val, len, std::back_inserter(subseq_kvs.vals));
      } else {
        // insert key into local PS
        request_handle_.insertKeyUnsafe(key, val);
      }


      // note threads to notify (send notifications outside the critical section)
      std::swap(thread_lists[i], queue.threads);
      queue.reset();
      request_handle_.unlockSingle(key);

      pos += len;
    }

    request_handle_.unlockAll(); // END CRITICAL SECTION

    // Forward the parameters for which a subsequent localize request has already arrived
    for(auto& sp : subsequent_localize_map) {
      auto& meta = sp.second.first->meta;
      auto& skvs = sp.second.second;
      this->Response(meta, skvs);
    }

    // Identify the threads we need to notify
    // TODO: could collect threads in a map earlier, while queuing
    std::unordered_map<std::pair<Customer*, int>, int, boost::hash<std::pair<Customer*, int>>> thread_responses;
    for (auto& threads : thread_lists) {
      for (size_t r=0; r!=threads.size(); ++r) {
        WaitingThread& thread = threads[r];
        thread_responses[std::make_pair(thread.customer, thread.ts)]++;
      }
    }

    // Notify threads about responses and send wake-up signal if appropriate
    for (auto& it : thread_responses) {
      Customer* customer = it.first.first;
      int timestamp = it.first.second;
      int numResponses = it.second;
      bool hasAllResponses = customer->AddResponse(timestamp, numResponses);
      if (hasAllResponses) { // send wake-up signal
        Message msg;
        msg.meta.control.cmd = Control::WAKE_SIGNAL;
        msg.meta.timestamp = timestamp;
        customer->Accept(std::move(msg));
      }
    }
  }


  /**
   * \brief Forward a pull/push request to another parameter server
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::Forward(int destination, const KVMeta& req, const KVPairs<Val>& fw) {
    Message msg;
    msg.meta.app_id      = this->obj_->app_id();
    msg.meta.customer_id = req.customer_id;
    msg.meta.request     = true;
    msg.meta.push        = req.push;
    msg.meta.head        = req.cmd;
    msg.meta.timestamp   = req.timestamp;
    msg.meta.recver      = Postoffice::Get()->ServerRankToID(destination);
    msg.meta.sender      = req.sender;
    if (fw.keys.size()) {
      msg.AddData(fw.keys);
      if (fw.vals.size()) {
        msg.AddData(fw.vals);
        if (fw.lens.size()) {
          msg.AddData(fw.lens);
        }
      }
    }

    Postoffice::Get()->van()->Send(msg);
  }

}  // namespace ps
#endif  // PS_COLOC_KV_SERVER_H_
