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
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <algorithm>
#include <atomic>
#include "ps/kv_app.h"
#include "ps/coloc_kv_worker.h"
#include "ps/addressbook.h"
#include <limits>
#include "ps/sync_manager.h"
#include "ps/sampling.h"

#include <iostream>
#include <sstream>

namespace ps {

// read management technique from stream
std::istream& operator>>(std::istream& in, MgmtTechniques& t) {
  std::string token; in >> token;
  if (token == "all") t = MgmtTechniques::ALL;
  else if (token == "replication_only") t = MgmtTechniques::REPLICATION_ONLY;
  else if (token == "relocation_only") t = MgmtTechniques::RELOCATION_ONLY;
  else { CHECK(false) << "Fatal! Unknown management technique selection " << token; }
  return in;
}
// write management technique to stream
std::ostream& operator<<(std::ostream &o, const MgmtTechniques& t) {
  switch(t) {
  case MgmtTechniques::ALL: return o << "all";
  case MgmtTechniques::REPLICATION_ONLY: return o << "replication_only";
  case MgmtTechniques::RELOCATION_ONLY: return o << "relocation_only";
  default: return o << "unkown";
  }
}

  /**
   * \brief A server node for maintaining key-value pairs
   */
  template <typename Val, typename Handle>
    class ColoKVServer : public KVServer<Val> {
    friend ColoKVWorker<Val, Handle>;
    friend SyncManager<Val, Handle>;
    template<typename V, typename W, typename S>
    friend class LocalSampling;
  public:

    /**
     * Construct a Lapse server with uniform value lengths
     */

    explicit ColoKVServer(size_t uniform_len) :
      ColoKVServer(std::vector<size_t> {uniform_len}) {}

    /**
     * \brief Construct a Lapse server
     */
    explicit ColoKVServer(std::vector<size_t> value_lengths) :
      postoffice_setup {Postoffice::Get()->ps_customer_id(0), Postoffice::Get()->num_channels()}, // first of all, set up node connections
      request_handle_(value_lengths),
      addressbook{request_handle_},
      my_rank{Postoffice::Get()->my_rank()},
      sync_managers_(Postoffice::Get()->num_channels()),
      workers(Postoffice::Get()->num_worker_threads(), nullptr),
      final_worker_clocks(Postoffice::Get()->num_worker_threads(), 0) {

      // initial parameter allocation
      for(Key key=0; key!=Postoffice::Get()->num_keys(); ++key) {
        if (addressbook.isManagedHere(key)) {
          request_handle_.insertKey(key, 0, true);
        }
      }

      // start the PS thread(s)
      using namespace std::placeholders;
      for (unsigned int c=0; c!=Postoffice::Get()->num_channels(); ++c) {
        this->customers_.push_back(new Customer(0, Postoffice::Get()->ps_customer_id(c),
                                                std::bind(&ColoKVServer<Val, Handle>::Process, this, c, _1)));
      }

      // start the sync manager thread(s)
      for (unsigned int i=0; i!=Postoffice::Get()->num_channels(); ++i) {
        sync_managers_[i].init(i, &request_handle_, this);
        sync_manager_threads_.push_back(std::thread (&SyncManager<Val,Handle>::thread, &sync_managers_[i]));
        std::string name = std::to_string(my_rank)+"-sm-"+Postoffice::Get()->fchannel(i);
        SET_THREAD_NAME((&sync_manager_threads_.back()), name.c_str());
      }

      lifetime.start();
    }

    ~ColoKVServer() {
      delete sampling_;
    }
    /**
    * \brief Retrieve the number of values per parameter key
    */
    inline size_t GetLen(const Key key) const {
      return request_handle_.get_len(key);
    }

    /**
     * \brief Forward a pull/push request to another parameter server
     */
    void Forward(int destination, const Meta& req, const KVPairs<Val>& fw);

    /**
     * \brief Shut down server (mostly writing statistics)
     */
    void shutdown() {
      lifetime.stop();
      // stop the sync manager thread(s)
      for (auto& rm : sync_managers_) {
        rm.stop();
      }
      for (auto& rmt : sync_manager_threads_) {
        rmt.join();
      }

      // terminate any sampling support activities
      if (sampling_ != nullptr) {
        sampling_->terminate();
      }

      // let the request handle print its own stats
      request_handle_.stats();

      // output per-server locality statistics
      ALOG(//pulls
            "server " << my_rank << ": " << num_pull_params << " parameters pulled, " <<
            100.0 * num_pull_params_local / num_pull_params << "% local (~" <<
            100.0 * request_handle_.get_num_pulls_to_replicas() / num_pull_params << "% to replicas) (" <<
            num_pull_ops << " ops, " << 100.0 * num_pull_ops_local / num_pull_ops << "% local)\n" <<
            // pushs
            "server " << my_rank << ": " << num_push_params << " parameters pushed, " <<
            100.0 * num_push_params_local / num_push_params << "% local (~" <<
            100.0 * request_handle_.get_num_pushs_to_replicas() / num_push_params << "% to replicas) (" <<
            num_push_ops << " ops, " << 100.0 * num_push_ops_local / num_push_ops << "% local)"
            );

      // write handle access statistics
      request_handle_.writeStats();

      // drop node connections and tear down the server object
      for (unsigned int c=0; c!=Postoffice::Get()->num_channels(); ++c) {
        Postoffice::Get()->Finalize(Postoffice::Get()->ps_customer_id(c), c==0);
      }
    }

    /**
     * \brief Enable support for sampling
     *
     *        If the sampling function of your application (`sample_key`) draws
     *        from a _continuous range_ of keys, you can pass the boundaries of
     *        this range with `min_key` (incl.) and `max_key` (excl.). This
     *        enables a more memory friendly implementation of local sampling.
     *
     */
    void enable_sampling_support(Key (*const sample_key)(), const Key min_key=0, const Key max_key=0) {
      auto& sst = Sampling<Val, ColoKVWorker<Val,Handle>, ColoKVServer<Val, Handle>>::scheme;
      switch (sst) {
      case SamplingScheme::Naive:
        sampling_ = new NaiveSampling<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>(sample_key, this);
        break;
      case SamplingScheme::Preloc:
        sampling_ = new PrelocSampling<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>(sample_key, this);
        break;
      case SamplingScheme::Pool:
        sampling_ = new PoolSampling<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>(sample_key, this);
        break;
      case SamplingScheme::Local:
        sampling_ = new LocalSampling<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>(sample_key, this, min_key, max_key);
        break;
      default:
        ALOG("Unkown sampling scheme '" << sst << "'. Aborting.");
        abort();
        break;
      }
    }

    // execute a barrier among all servers
    void Barrier() {
      Postoffice::Get()->Barrier(0, kServerGroup) ;
    }

    // add system options to an application
    static void AddSystemOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sys.zmq_threads", po::value<int>(&Postoffice::Get()->num_network_threads_)->default_value(3), "number of ZMQ threads")
        ("sys.techniques", po::value<MgmtTechniques>(&Postoffice::Get()->management_techniques_)->default_value(MgmtTechniques::ALL), "Which management techniques to use. Options: 'all' (combine relocation and replication), 'replication_only', 'relocation_only'")
        ("sys.time_intent_actions", po::value<bool>(&Postoffice::Get()->time_intent_actions_)->default_value(true), "Whether to time intent actions. If disabled, the system acts on intent with the next sync. Otherwise, the system will include intent in syncs only when it thinks it is appropriate.")
        ("sys.location_caches", po::value<bool>(&Postoffice::Get()->location_caches_)->default_value(true), "whether or not to use location caches")
        ("sys.channels", po::value<int>(&Postoffice::Get()->num_channels_)->default_value(4), "the number of communication channels (per process, we start one PS thread and one replica manager thread per channel). The number of channels should be a power of 2.")
        ("sys.trace.keys", po::value<std::string>(&Postoffice::Get()->traced_keys_list_)->default_value(""), "Comma-separated list of keys that should be traced. Relocation, replication, and intent will be traced for these keys. Syntax example: '12,3421,37892'. Also possible: 'all' to trace all keys and 'random-10-seed-7-range-0-100' to trace a random sample (seed 7) of 10 keys in the range [0, 100). To enable tracing, compile with flag 'PS_TRACE_KEYS=1'")
        ("sys.stats.out", po::value<std::string>(&Postoffice::Get()->stats_output_folder_)->default_value(""), "Where to write statistics (e.g., key traces or locality statistics)")
        ;

      // replication options
      SyncManager<Val,Handle>::AddSyncOptions(options);

      // sampling options
      Sampling<Val,ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>::AddSamplingOptions(options);
    }

    // store a pointer to each worker (so that the server can access them if needed)
    void registerWorker(int customer_id, ColoKVWorker<Val, Handle>* worker) {
      workers[customer_id] = worker;
      ++workers_num_registered;
    }

    // note that a worker has exited
    void deregisterWorker(int customer_id) {
      final_worker_clocks[customer_id] = workers[customer_id]->currentClock(); // store the final clock of the worker
      workers[customer_id] = nullptr;
    }

    // wait until all workers have been started
    void ensureAllWorkersAreRegistered() {
      while (workers_num_registered < Postoffice::Get()->num_worker_threads()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }

    // obtain a pointer to the worker with `customer_id`
    ColoKVWorker<Val, Handle>* getWorker(const int customer_id) {
      return workers[customer_id];
    }

    // obtain the current clocks of all local workers
    std::vector<Clock> getWorkerClocks() {
      ensureAllWorkersAreRegistered();

      std::vector<Clock> clocks (Postoffice::Get()->num_worker_threads(), 0);
      for (size_t w=0; w!=Postoffice::Get()->num_worker_threads(); ++w) {
        if (workers[w] != nullptr) {
          clocks[w] = workers[w]->currentClock();
        } else {
          // worker has exited already
          clocks[w] = WORKER_FINISHED;
        }
      }
      return clocks;
    }

    // obtain the total number of clock ticks by local workers
    Clock getTotalTicks() {
      Clock total_ticks = 0;
      for (size_t w=0; w!=Postoffice::Get()->num_worker_threads(); ++w) {
        if (workers[w] != nullptr) {
          total_ticks += workers[w]->currentClock();
        } else {
          total_ticks += final_worker_clocks[w];
        }
      }
      return total_ticks;
    }

    // pushes an update into the local store if possible; returns directions otherwise
    inline int localPushOrDirections(const Key key, const Val* val, const bool set) {
      request_handle_.lockSingle(key);

      auto direction = request_handle_.attemptLocalPushUnsafe(key, val, set, true) ?
        LOCAL :
        addressbook.getDirectionsUnsafe(key, true); // use location cache

      request_handle_.unlockSingle(key);
      return direction;
    }

    // pulls a parameter value from the local store if possible; returns directions otherwise
    inline int localPullOrDirections(const Key key, Val* val) {
      request_handle_.lockSingle(key);

      auto direction = request_handle_.attemptLocalPullUnsafe(key, val, true) ?
        LOCAL :
        addressbook.getDirectionsUnsafe(key, true); // use location cache

      request_handle_.unlockSingle(key);
      return direction;
    }

    // output selected management techniques as string
    std::string management_techniques_str() {
      std::stringstream ss;
      ss << Postoffice::Get()->management_techniques();
      return ss.str();
    }

  private:

    // Send out forward messages (to other servers)
    void sendForwards(std::vector<KVPairs<Val>>& forwards, const Meta& meta) {

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

    PostofficeSetup postoffice_setup; // helper object to set up postoffice before anything else happens
    Handle request_handle_; // hides the request_handle_ of KVServer
    bool Process(const unsigned int channel, const Message& msg); // we re-define Process to use the correct request_handle_
    void ProcessPushPullRequest(const Message& msg);
    Addressbook<Handle> addressbook;
    int my_rank; // the rank of this server

    // customers for PS threads (replaces `KVServer::obj_`)
    std::vector<Customer*> customers_;

    // replica management
    std::vector<SyncManager<Val,Handle>> sync_managers_;
    std::vector<std::thread>   sync_manager_threads_;

    // sampling management
    Sampling<Val, ColoKVWorker<Val, Handle>, ColoKVServer<Val, Handle>>* sampling_=nullptr;

    // maintain a connection to the worker objects
    std::vector<ColoKVWorker<Val, Handle>*> workers;
    std::atomic<size_t> workers_num_registered { 0 };

    // store the last clocks of workers (so that we know this clock after the worker has deregistered)
    std::vector<Clock> final_worker_clocks;

    util::Stopwatch lifetime; // lifetime of this server

    long num_pull_ops=0, num_pull_ops_local=0, num_push_ops=0, num_push_ops_local=0;
    long num_pull_params=0, num_pull_params_local=0, num_push_params=0, num_push_params_local=0;
    long num_pull_params_rep=0, num_push_params_rep=0;
  };


  /**
   * \brief Process a request to a parameter server
   */
  template <typename Val, typename Handle>
  bool ColoKVServer<Val, Handle>::Process(const unsigned int channel, const Message& msg) {
    assert(msg.meta.channel != Meta::kEmpty); // every PS-processed message should have channel information
    assert(static_cast<int>(channel) == msg.meta.channel); // make sure the message has been routed to the correct channel

    if (msg.meta.head == Control::SYNC) {
      sync_managers_[msg.meta.channel].ProcessSyncMessage(msg, false);
    } else if (msg.meta.head == Control::SYNC_FORWARD) {
      sync_managers_[msg.meta.channel].ProcessSyncMessage(msg, true);
    } else if (msg.meta.request) { // process a push or a pull requests
      ProcessPushPullRequest(msg);
    }  else {
      LL << "FATAL: unknown type of message in server thread";
    }
    return false;
  }

  /**
   * \brief Process a regular (i.e., push/pull) request to a parameter server
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::ProcessPushPullRequest(const Message& msg) {
    // parse request key/value data
    KVPairs<Val> data;
    data.keys = msg.data[0];
    if (msg.meta.push) {
      data.vals = msg.data[1];
    }

    // prepare response key/value data
    KVPairs<Val> res;
    if (!msg.meta.push) {
      // don't know exactly how many keys will be answered locally
      // overallocate (largest possible case: all keys are local)
      res.vals.resize(request_handle_.get_total_len(data.keys), 0, false);
    }


    // prepare a couple of forward messages to other servers
    std::vector<KVPairs<Val>> forwards(Postoffice::Get()->num_servers());

    uint numLocal = 0;
    size_t len = 0, data_pos = 0, res_pos = 0;
    // for each key, attempt a local pull(/push). forward the keys that are not local

    for (size_t i = 0; i < data.keys.size(); ++i) {
      Key key = data.keys[i];
      len = request_handle_.get_len(key);
      bool local;
      if (msg.meta.push) // push request
        local = request_handle_.attemptLocalPush(key, data.vals.begin()+data_pos, msg.meta.set, false);
      else // pull request
        local = request_handle_.attemptLocalPull(key, res.vals.data()+res_pos, false);

      if (local) {
        if (!msg.meta.push) { // add key to reply for pull requests
          res.keys.push_back(key);
          res_pos += len;
        }
        ++numLocal;
      } else {
        auto destination = addressbook.getDirections(key, false); // don't use location cache for forwards
        forwards[destination].keys.push_back(key);
        if (msg.meta.push) { // forward the "to push" values to the responsible server
          std::copy_n(data.vals.begin() + data_pos, len, std::back_inserter(forwards[destination].vals));
        }
      }
      data_pos += len;
    }

    // if anything was or will be answered locally, send a response to the original sender
    // (either right now or later)
    if (numLocal != 0) {
      if(msg.meta.push) {
        // for a push, we send back the number of locally processed keys
        res.vals.resize(0);
        res.keys.resize(1);
        res.keys[0] = numLocal;
      } else {
        res.vals.resize(request_handle_.get_total_len(res.keys));
        // need to do this also for the direct message
      }

      this->Response(msg.meta, res);
    }

    // if any key was not answered locally, send forward message(s)
    if (numLocal < data.keys.size()) {
      sendForwards(forwards, msg.meta);
    }
  }

  /**
   * \brief Forward a pull/push request to another parameter server
   */
  template <typename Val, typename Handle>
    void ColoKVServer<Val, Handle>::Forward(int destination, const Meta& req, const KVPairs<Val>& fw) {
    Message msg;
    msg.meta.app_id      = 0;
    msg.meta.customer_id = req.customer_id;
    msg.meta.channel     = req.channel;
    msg.meta.request     = true;
    msg.meta.push        = req.push;
    msg.meta.set         = req.set;
    msg.meta.head        = req.head;
    msg.meta.timestamp   = req.timestamp;
    msg.meta.recver      = Postoffice::Get()->ServerRankToID(destination);
    msg.meta.sender      = req.sender;
    if (fw.keys.size()) {
      msg.AddData(fw.keys);
      if (fw.vals.size()) {
        msg.AddData(fw.vals);
      }
    }

    Postoffice::Get()->van()->Send(msg);
  }

}  // namespace ps
#endif  // PS_COLOC_KV_SERVER_H_
