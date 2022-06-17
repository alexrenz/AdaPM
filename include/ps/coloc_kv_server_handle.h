/**
 *  Copyright (c) 2018 by Contributors
 */
#ifndef PS_COLOC_SERVER_HANDLE_
#define PS_COLOC_SERVER_HANDLE_
#include <algorithm>
#include <utility>
#include <vector>
#include "ps/coloc_kv_worker.h"
#include "ps/coloc_kv_server.h"
#include <thread>
#include <mutex>
#include <memory>
#include <regex>

#include <iostream>
#include <sstream>



/*

  Default server handle for co-located parameter servers.
  Treats push's as increments.

  There are multiple implementations of the backend data structure.
  Currently supported are:
  (1) std::unordered_map with a lock array (PS_BACKEND_STD_UNORDERED_LOCKS=1)
       - For each key, there is one lock. But each lock is for multiple keys
       - For pushs/pulls, the appropriate lock is acquired
       - For parameter moves (when we need to add/remove parameters to the map),
         we acquire all locks. (LOCK_ALL)
  (2) std::vector with a lock array (PS_BACKEND_VECTOR_LOCKS=14) [DEFAULT]
       - As (1), but for a parameter move, we don't lock the entire data structure.
         We acquire only the lock for the moved parameter. (LOCK_SINGLE)
  (3) std::vector without locks (PS_BACKEND_VECTOR=10)
       - As (2), but without locks. This is useful when consistency is guaranteed
         in another way. For example when shared memory access to parameters is disabled.
         In this case, consistency is guaranteed because only the server thread accesses
         local parameters.

   The desired backend data structure can be specified with the compilation flag
   PS_BACKEND in cmake. E.g, to compile with the unordered map as backend data
   structure, run cmake as follows: `cmake -S . -B build -DPS_BACKEND=1`

*/


// PS back-end data structure variants
#define PS_BACKEND_STD_UNORDERED_LOCKS 1
#define PS_BACKEND_VECTOR 10
#define PS_BACKEND_VECTOR_LOCKS 14

#define LOCK_ALL 1
#define LOCK_SINGLE 0

// default back-end data structure: vector with locks (and lock-single locking strategy)
#ifndef PS_BACKEND
#define PS_BACKEND PS_BACKEND_VECTOR_LOCKS
#endif

// Use locks?
#define PS_BACKEND_LOCKS PS_BACKEND == PS_BACKEND_STD_UNORDERED_LOCKS || PS_BACKEND == PS_BACKEND_VECTOR_LOCKS
// default number of locks
#ifndef PS_BACKEND_NUM_LOCKS
#define PS_BACKEND_NUM_LOCKS 16384
#endif

// if we have a vector backend structure, we use LOCK_SINGLE by default, otherwise we use LOCK_ALL
#ifndef PS_BACKEND_LOCK_STRATEGY
#define PS_BACKEND_LOCK_STRATEGY PS_BACKEND != PS_BACKEND_VECTOR_LOCKS
#endif

// trace relocations, replications, and intent for some keys
#ifndef PS_TRACE_KEYS
#define PS_TRACE_KEYS 0
#endif

// backend locality statistics
#ifndef PS_LOCALITY_STATS
#define PS_LOCALITY_STATS 0
#endif
namespace ps {

  // For key tracing
  enum class Event {ALLOC, DEALLOC, REPLICA_SETUP, REPLICA_DROP, INTENT_START, INTENT_STOP};
  struct Trace {
    Trace (std::chrono::time_point<std::chrono::high_resolution_clock> t,
           Key k, Event e): time{t}, key{k}, event{e} {};
    std::chrono::time_point<std::chrono::high_resolution_clock> time;
    Key key;
    Event event;
  };
  std::ostream& operator<<(std::ostream &o, const Event& e) {
    switch(e) {
    case Event::ALLOC: return o << "alloc";
    case Event::DEALLOC: return o << "dealloc";
    case Event::REPLICA_SETUP: return o << "replica_setup";
    case Event::REPLICA_DROP: return o << "replica_drop";
    case Event::INTENT_START: return o << "intent_start";
    case Event::INTENT_STOP: return o << "intent_stop";
    default: return o << "unknown";
    }
  }

  struct Intent {
    Intent(): start{std::numeric_limits<Clock>::max()}, end{-1} {}
    Intent(Clock s, Clock e): start{s}, end{e} {}
    Clock start;
    Clock end;
  };

  std::ostream& operator<<(std::ostream& os, const Intent i) {
    std::stringstream ss;
    ss << "[" << i.start << "-" << i.end << "]";
    os << ss.str();
    return os;
  }


  /* A parameter in the local parameter server */
  template <typename Val>
    struct Parameter {
      // Constructor: normal feature, no inital value
    Parameter(size_t len):
      val(len),
      local_intents{} {}
      // Constructor: normal feature, initial value given
    Parameter(Val* init_val, size_t len):
      val(init_val, init_val+len),
      local_intents{} {}

      // the parameter value
      std::vector<Val> val;

      // for replicas: the version of the last sync
      std::vector<Val> sync_state;

      // indicates whether this parameter is a replica
      bool replica = false;

      // indicates whether this parameter was updated since the last sync
      bool updated = false;

      // number of updates to (the main copy) of this parameter
      Version version = 0;

      Version relocation_counter = 0;

      // which workers require this replica until which clock?
      std::unordered_map<int, Clock> local_intents;
    };

template <typename Val>
struct DefaultColoServerHandle {
public:

  // Uniform parameter lengths: each parameter value has the same length
  DefaultColoServerHandle(size_t uniform_len): DefaultColoServerHandle(std::vector<size_t> {uniform_len}) { }

  // Non-uniform parameter lengths: each parameter can have its own value length
  DefaultColoServerHandle(const std::vector<size_t>& value_lengths):
    value_lengths {value_lengths},
    replicas     (Postoffice::Get()->num_channels()),
    replicas_mus (Postoffice::Get()->num_channels()),
    p_num_channels_(std::log2(Postoffice::Get()->num_channels())) {

    long store_max = Postoffice::Get()->num_keys();
    ADLOG("Creating handle for " << store_max << " keys with " <<
          (value_lengths.size() == 1 ? "uniform" : "non-uniform") << " lengths ");

#if PS_BACKEND == PS_BACKEND_STD_UNORDERED_LOCKS
    ALOG("Handle data structure: std::unordered_map with " << mu_.size() << " locks");
    store.reserve(store_max);

#elif PS_BACKEND == PS_BACKEND_VECTOR
    ALOG("Handle data structure: vector<unique_ptr<Parameter>> (no locks)");
    store.resize(store_max);

#elif PS_BACKEND == PS_BACKEND_VECTOR_LOCKS
    ALOG("Handle data structure: vector<unique_ptr<Parameter>> with " << mu_.size() << " locks");
    store.resize(store_max);

#else
    ALOG("Handle data structure: std::unordered");

#endif

#if PS_BACKEND_LOCKS
#if PS_BACKEND_LOCK_STRATEGY == LOCK_ALL
    ALOG("Locking strategy: LOCK_ALL");
#elif PS_BACKEND_LOCK_STRATEGY == LOCK_SINGLE
    ALOG("Locking strategy: LOCK_SINGLE");
#endif
#endif

    my_rank = Postoffice::Get()->my_rank(); // to make sure it is available at shut down

    // check that the number of channels is a power of 2 (which we require for our channel assignment method)
    if ((Postoffice::Get()->num_channels() & (Postoffice::Get()->num_channels() - 1)) != 0) {
      ALOG("The number of channels should be a power of 2, but you have specified " << Postoffice::Get()->num_channels());
      abort();
    }

    // capture detailed locality statistics
#if PS_LOCALITY_STATS
    ADLOG("Capture locality statistics for " << store_max << " keys");
    num_accesses.resize(store_max, 0);
    num_accesses_local.resize(store_max, 0);
#endif

    // trace relocations, replications, and intents for some keys
#if PS_TRACE_KEYS
    std::istringstream keyList(Postoffice::Get()->get_traced_keys());
    std::string key;
    while (std::getline(keyList, key, ',')) {
      if(key.find("random") != std::string::npos) {
        // mark a number of randomly drawn keys for tracing
        std::regex rgx("^random-([0-9]+)-seed-([0-9]+)-range-([0-9]+)-([0-9]+)$");
        std::smatch matches;
        if(std::regex_search(key, matches, rgx)) {
          unsigned long num_keys_to_add = std::stol(matches[1]);
          long seed = std::stol(matches[2]);
          Key minkey = std::stol(matches[3]);
          Key maxkey = std::stol(matches[4]);
          if (maxkey == 0) maxkey = store_max;

          // randomly draw the specified number of keys
          std::uniform_int_distribution<Key> key_dist (minkey, maxkey-1);
          std::mt19937 rng (seed);
          std::unordered_set<Key> random_keys {};
          while (random_keys.size() != num_keys_to_add) {
            random_keys.insert(key_dist(rng));
          }

          // mark the keys for tracing
          traced_keys.insert(random_keys.begin(), random_keys.end());

          if (my_rank == 0) ALOG("Trace " << num_keys_to_add << " random keys (seed " << seed << ") in range [" << minkey << ", " << maxkey << "): " << str(random_keys));
        } else {
          ALOG("Don't understand how to trace '" << key << ". Abort");
          abort();
        }
      } else if (key.find("all") != std::string::npos) {
        for (Key k=0; k!=static_cast<Key>(store_max); ++k) {
          traced_keys.insert(k);
        }

      } else {
        // mark a single key for tracing
        traced_keys.insert(std::stol(key));
      }
    }
    if (my_rank == 0) ALOG("Trace " << traced_keys.size() << " keys: " << str(traced_keys));
#endif

    // clean stats folder (if any specified)
    if (!Postoffice::Get()->get_stats_output_folder().empty() && my_rank == 0) {
      auto folder = Postoffice::Get()->get_stats_output_folder();
      auto cmd = std::string("mkdir -p ") + folder + std::string(" && rm ") + folder + std::string("/*.tsv");
      if (std::system(cmd.c_str()) == 0) {
        ALOG("Cleaned stats output folder " << folder);
      } else {
        ALOG("Failed to clean stats output folder " << folder);
      }
    }
  }

  ~DefaultColoServerHandle(){
  }

#if PS_BACKEND_LOCKS
  inline size_t lockForKey(Key key) {
    return key % mu_.size();
  }
#endif

  /**
   * \brief Inserts a key into the local data structure
   */
  void insertKey(Key key, Val* val=0, bool initial_alloc=false) {
    lockAll();
    lockSingle(key);
    insertKeyUnsafe(key, val);
    unlockSingle(key);
    unlockAll();

    if (initial_alloc) {
      trace_key(key, Event::ALLOC);
    }
  }

  /**
   * \brief Inserts a key into the local data structure.
   *
   *        Warning: this method is not thread safe.
   */
  inline void insertKeyUnsafe(Key key, Val* val=0) {
    // add key to local store
#if PS_BACKEND < 10 // map
    if (val == 0) { // default value
      store[key] = Parameter<Val>(get_len(key)); // use default value for Val (typically 0)
    } else {
      store[key] = Parameter<Val>(val, get_len(key));
    }
# else
    if (val == 0) { // default value
      store[key] = std::make_unique<Parameter<Val>>(get_len(key));
    } else {
      store[key] = std::make_unique<Parameter<Val>>(val, get_len(key));
    }
#endif
  }

  /**
   * \brief Removes a key from the local data structure
   */
  inline void removeKey(Key key) {
    lockAll();
    lockSingle(key);
    removeKeyUnsafe(key);
    unlockSingle(key);
    unlockAll();
  }

  /**
   * \brief Removes a key from the local data structure (without synchronization)
   *
   *        Warning: this method is not thread safe.
   */
  inline void removeKeyUnsafe(Key key) {
#if PS_BACKEND < 10 // map
    store.erase(key);
#else
    store[key].reset();
#endif
  }

  /**
   * \brief Attempt to push a the value for a key into this parameter server
   *
   * @param key the key
   * @param val a pointer to the corresponding values
   * @return true if the key is local (and therefore, was pushed), false otherwise
   */
  inline bool attemptLocalPush(const Key key, const Val* val, const bool set, bool stats=true) {
    if (isNonLocal_noLock(key)) { return false; } // try to return early

#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return attemptLocalPushUnsafe(key, val, set, stats);
  }

  /**
   * \brief Attempt to push a the value for a key into this parameter server
   *
   *        Warning: this method is not thread safe.
   */
  inline bool attemptLocalPushUnsafe(const Key key, const Val* val, const bool set, bool stats=true) {
    // attempt to push the value for this key into the local store
#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end()) {
      return false;
    } else {
      auto& param = search->second;
#else
    if (store[key] == nullptr) {
      return false;
    } else {
      auto& param = *store[key];
#endif

      // don't use a replica that isn't properly set up yet
      if (param.version == -1) {
        return false;
      }

      mergeValue(param.val, val, set, get_len(key));

      // if this parameter is replicated, we additionally note the updates separately,
      // so that we can send delta updates to the server later
      if (param.replica) {
        param.updated = true;
        ++num_pushs_to_replicas;

        // set operations do not return meaningful results in the presence of replicas.
        // for now, let's abort when a set operation is used on a replica
        assert(!set);
      } else {
        // if this is the main copy, increase the version counter
        ++param.version;
        assert(param.version != 0); // let's hear about overflows for now
      }

      return true;
    }
  }

  /**
   * \brief Merge a value `merge` into an existing value `target`
   */
  template<typename V1, typename V2>
    inline void mergeValue(V1& target, V2& merge, const bool set, const size_t len) {
    if (set) { // set value
      for(uint i=0; i!=len; ++i) {
        target[i] = merge[i];
      }
    } else { // push (i.e., add) value
      for(uint i=0; i!=len; ++i) {
        target[i] += merge[i];
      }
    }
  }

  /**
   * \brief Attempt to pull the value of a key from this parameter server
   *
   * @param key the key
   * @param vals the array to put the value into
   * @return true if the key is local (and therefore, the current value is in vals), false otherwise
   */
inline bool attemptLocalPull(const Key key, Val* val, const bool stats=true) {
    if (isNonLocal_noLock(key)) { return false; } // try to return early

#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return attemptLocalPullUnsafe(key, val, stats);
  }

  /**
   * \brief Attempt to pull the value of a key from this parameter server
   *
   *        Warning: this method is not thread safe.
   */
  inline bool attemptLocalPullUnsafe(const Key key, Val* val, const bool stats=true) {
#if PS_LOCALITY_STATS
    if (stats) num_accesses[key]++;
#endif

#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end()) {
      return false; // this key is not local
    } else {
      auto& param = search->second;
#else
    if (store[key] == nullptr) {
      return false; // this key is not local
    } else {
      auto& param = (*store[key]);
#endif

      // don't use a replica that isn't properly set up yet
      if (param.version == -1) {
        return false;
      }

      // key is local

      // the actual pull: copy the stored value to the passed location
      std::copy_n(begin(param.val), get_len(key), val);

      // stats
      if (param.replica) {
        ++num_pulls_to_replicas;
      }

      // stats
#if PS_LOCALITY_STATS
      if (stats) num_accesses_local[key]++;
#endif

      return true;
    }
  }

  /**
   * \brief Register a set of new intents for one key.
   *        This stores the intents in the handle and creates the parameter object if necessary
   */
  void registerNewIntentsForKeyUnsafe(const Key key,
                                      std::vector<WorkerClockPair>& new_intents) {

    // insert key if necessary
    bool inserted = false;
#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end())
#else
    if(store[key] == nullptr)
#endif
      {
        insertKeyUnsafe(key);
        inserted = true;
      }

    // mark as replica
#if PS_BACKEND < 10 // map
    auto& param = store[key];
#else
    auto& param = (*store[key]);
#endif

#if PS_TRACE_KEYS
    if (param.local_intents.size() == 0) {
      trace_key(key, Event::INTENT_START);
    }
#endif

    for (auto& new_intent : new_intents) {

      // is there intent by this worker already?
      auto search = param.local_intents.find(new_intent.customer_id);

      if (search == param.local_intents.end()) { // if there is no intent yet, add the new one one
        param.local_intents.insert({new_intent.customer_id, new_intent.end});
      } else { // o/w, we merge the two intents: keep the later end clock
        search->second = std::max(search->second, new_intent.end);
      }
    }

    // set up parameter
    if (inserted) {
      param.replica = true;
      param.version = -1; // mark that this replica cannot be used yet (until the sync refreshes it)

      rememberReplica(key);
    }
  }

  /**
   * \brief Does at least one worker on this node currently have intent for `key`?
   *
   *        While checking, this method cleans up any intents that are outdated
   *        because the corresponding worker has already passed the `end` clock
   *        of the intent.
   *
   */
  bool hasLocalIntentUnsafe(const Key key,
                            const std::vector<Clock>& worker_clocks) {
#if PS_BACKEND < 10 // map
    auto& param = store[key];
    assert(store[key] != nullptr);
#else
    auto& param = (*store[key]);
#endif

    // fast exit
    // (note: fast exit is important for the correctness of the key tracing below (INTENT_STOP))
    if (param.local_intents.size() == 0) {
      return false;
    }

    // iterate the current intents
    auto it = param.local_intents.begin();
    while (it != param.local_intents.end()) {
      auto& customer_id = it->first;
      auto& intent_end = it->second;

      if (worker_clocks[customer_id] < intent_end) {
        // this is a current intent
        return true;
        // note: if we were to continue the loop, we would increase the iterator here: ++it;
      } else {
        // this intent is outdated. delete it, then continue to iterate
        it = param.local_intents.erase(it); // erase() returns an iterator to the next element
      }
    }

    // the had intent above (otherwise we would have exited quickly), and now the node doesn't
    trace_key(key, Event::INTENT_STOP);

    // we found no current intent
    return false;
  }

  // quick check for local intent (without cleanup)
  bool hasLocalIntent_noCleanup_Unsafe(Parameter<Val>& param) {
    return param.local_intents.size() != 0;
  }

  /**
   * \brief Visit a replica for sync and replica deletion, i.e.:
   *            (1) extract accumulated updates if there are any
   *            (2) drop the replica if no worker needs it anymore
   *
   *        Returns a tuple <dropped, version>, indicating
   *            (1) whether the method has dropped the replica (because it isn't needed anymore)
   *            (2) which version the replica has
   *
   *        Extracts updates selectively, depending on `threshold`:
   *        - threshold=-1: always extract an update (also if it is all zero)
   *        - threshold=0: extract update if the parameter was updated
   *        - threshold>0: extract update if its norm exceeds the given threshold,
   *                       i.e., if norm(updates)>threshold; if we drop the replica,
   *                       any update will be extracted, regardless of the threshold
   */
  std::tuple<bool, bool, Version>
    readAndPotentiallyDropReplica(const Key key, KVPairs<Val>& message,
                                  const std::vector<Clock>& worker_clocks,
                                  const double threshold=-1) {
#if PS_BACKEND_LOCKS
     std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

#if PS_BACKEND < 10 // map
     auto& param = store[key];
#else // vector
     auto& param = (*store[key]);
#endif

     assert(param.replica); // the parameter has to be a replica

     // decide whether this replica should be dropped
     // (we will drop it if none of the local worker needs it anymore)
     const bool localIntent = hasLocalIntentUnsafe(key, worker_clocks);

     // determine the current version of the replica (so we can return it later)
     const auto replica_version = param.version;

     // extract update if desired
     bool extracted_an_update = false;
     bool drop = !localIntent;
     if ((param.version != -1) &&  // no extraction if replica isn't set up yet
         ((threshold == -1) || // always extract
          (threshold == 0 && param.updated) || // extract only if updated
          (threshold > 0 && param.updated && // extract if update exceeds threshold
           (diffl2norm(param.val, param.sync_state) >= threshold || drop)))) {

       extracted_an_update = true;

       // copy out the updates
       for (size_t k=0; k!=get_len(key); ++k) {
         message.vals.push_back(param.val[k] - param.sync_state[k]);
         param.sync_state[k] = param.val[k];
        }
       param.updated = false;

       // We raise the version counter to v+1 to prevent unnecessary refreshes.
       // We can do so safely because we know that the main copy will increase
       // its version counter to v+1 for the update that we just extracted. When
       // there are updates by other nodes, the main version will be >v+1 and
       // this node's replica will be refreshed.
       ++param.version;
     }

     bool refresh;
     if (localIntent) {
       refresh = true;
     } else {
       assert(!param.updated); // there should not be any update left
       removeKeyUnsafe(key);
       forgetReplica(key);
       refresh = false;
       trace_key(key, Event::REPLICA_DROP);
     }

     return std::make_tuple(refresh, extracted_an_update, replica_version);
   }

  /**
   * \brief Does this node currently hold own `key`?
   */
  bool isOwnerUnsafe(const Key key) {
#if PS_BACKEND < 10
    auto search = store.find(key);
    return search != store.end() && !search->second.replica;
#else
    return store[key] != nullptr && !store[key]->replica;
#endif
  }

   /**
    * \brief Touch an owned key for several (all optional) actions:
    *       (1) [optional] Write an external update into the main replica
    *       (2) [optional] Copy out the latest value
    *       (3) [optional] Remove the parameter from the local store (for relocation to another node)
    *
    *        In more detail:
    *        (1) The method writes an external update if `update`!=nullptr.
    *        (2) The method copies out the latest value (for a replica refresh) if
    *             (a) a refresh is desired (indicated by `copy`==true)
    *             (b) there are updates that the requester does not yet have
    *        (3) The method removes the parameter from the local store if `relocate`==true
    *
    *        Returns a pair <bool a, version b> that indicates
    *         (a) whether the latest value was extracted and
    *         (b) what the latest version of the main copy is
    *         (c) the relocation counter oft he parameter
    */
  std::tuple<bool, Version, Version>
    writeCopyDropOwnedKeyUnsafe(const Key key, const Val* update,
                                const bool copy, KVPairs<Val>& response,
                                const Version replica_ver, const bool relocate) {
#if PS_BACKEND < 10 // map
     auto& param = store[key];
#else
     auto& param = (*store[key]);
#endif

     assert(!param.replica);
     assert(replica_ver <= param.version);

     // does the main copy have updates that the requester hasn't seen yet
     // note: important to read main version before the potential update below
     bool have_unsynced_updates = replica_ver < param.version;

     // merge in the external update
     if (update != nullptr) {
       mergeValue(param.val, update, false, get_len(key));
       ++param.version;
       assert(param.version != 0); // let's hear about overflows for now
     }

     // copy out the latest value
     bool copied = false;
     if (copy && have_unsynced_updates) {
       copied = true;
       std::copy_n(std::begin(param.val), get_len(key), std::back_inserter(response.vals));
     }

     // if we relocate, drop the parameter from this node's store
     auto ver = param.version;
     auto relocation_counter = param.relocation_counter;
     if (relocate) {
       assert(copy); // if we drop a parameter, update extraction should be enabled
       bool localIntent = hasLocalIntent_noCleanup_Unsafe(param);
       if (Postoffice::Get()->management_techniques() == MgmtTechniques::RELOCATION_ONLY &&
           localIntent) {
         // Special case: only relocation is possible + there still is local
         // intent. We keep the parameter object around (as it stores the local
         // intent) and localize the parameter again in the next sync round (if
         // there still is intent then)

         // reset val
         std::fill(begin(param.val), end(param.val), 0);
         param.version = -1;
         param.updated = false;

         // include in next sync round
         param.replica = true;
         rememberReplica(key);
       } else {
         // Normal case: by design, we know that there is no local intent
         // anymore (because we relocate only when there is no other intent). So
         // we can safely drop the parameter.
         assert(!localIntent);
         removeKeyUnsafe(key);
       }
       trace_key(key, Event::DEALLOC);
     }

     return std::make_tuple(copied, ver, relocation_counter);
   }

   /**
    * \brief Obtain a list of replicas that this node currently holds
    */
   std::vector<Key> currentReplicas(const unsigned int channel) {
     std::lock_guard<std::mutex> lk(replicas_mus[channel]);
     std::vector<Key> current_replicas (replicas[channel].begin(), replicas[channel].end());
     return current_replicas;
   }

   /**
    * \brief Touch a replicated parameter for multiple (optional) actions:
    *           (1) [optional] Refresh replica, i.e., write an external update
    *           (2) [optional] Upgrade the replica to an owned parameter
    *
    *        Adds updates that were accumulated since the synchronization mechanism
    *        last read the replica.
    */
   void refreshUpgradeReplicaUnsafe(const Key key, const Val* state, const Version ver,
                                    const bool upgrade_to_owned, const Version relocation_counter) {
#if PS_BACKEND < 10 // map
     auto search = store.find(key);
     assert (search != store.end());
     auto& param = search->second;
#else
     assert (store[key] != nullptr); // the replica should still be in store
     auto& param = (*store[key]);
#endif
     assert(param.replica);

     // update the replica
     if (state != nullptr) {
       if (param.sync_state.size() == 0) {
         // proceed without previous sync state
         // (we have this special case to omit the `sync_state` allocation for relocations)
         for (size_t j=0; j!=get_len(key); ++j) {
           param.val[j] = state[j] + param.val[j];
         }
       } else {
         for (size_t j=0; j!=get_len(key); ++j) {
           param.val[j] = state[j] + param.val[j] - param.sync_state[j];
         }
       }

       // write last sync state (which is not necessary if this is a relocation)
       if (!upgrade_to_owned) {
         if (param.sync_state.size() == 0) {
           param.sync_state.resize(get_len(key));
         }

         // update sync state
         memcpy(param.sync_state.data(), state, get_len(key)*sizeof(Val));
       }

#if PS_TRACE_KEYS
       if (param.version == -1) {
         trace_key(key, Event::REPLICA_SETUP);
       }
#endif

       param.version = ver;
     }

     // relocation: this replica now becomes a parameter that is owned by this node
     if (upgrade_to_owned) {
       param.replica = false;
       param.relocation_counter = relocation_counter;
       forgetReplica(key);

       // if the replica has been updated in the mean time, these updates lead to a version raise
       if (param.updated) {
         ++param.version;
         param.updated = false;
       }

#if PS_TRACE_KEYS
       if (param.version != -1) {
         trace_key(key, Event::REPLICA_DROP);
       }
       trace_key(key, Event::ALLOC);
#endif
     }
   }

  // output stats about the number of local parameters and replicas
  void stats () {
    size_t in_store = 0;
    size_t replica = 0;
    for(auto& param : store) {
#if PS_BACKEND < 10 // map
      ++in_store;
      if (param.second.replica) ++replica;
#else
      if(param != nullptr) {
        ++in_store;
        if (param->replica) ++replica;
      }
#endif
    }

    ALOG("RH@" << Postoffice::Get()->my_rank() << ": " << in_store << " in store, " << replica << " replicas");
  }


  /*
    LOCK_ALL strategy
    If we move a parameter (i.e., take it out of the store or put it into the store),
    we lock the entire data structure
  */

  /**
   * \brief Acquire locks for all keys (for LOCK_ALL strategy)
   *        Does nothing if we use LOCK_SINGLE
   */
  inline void lockAll() {
#if PS_BACKEND_LOCKS
#if PS_BACKEND_LOCK_STRATEGY == LOCK_ALL
    for(size_t i=0; i!=mu_.size(); ++i) {
      mu_[i].lock();
    }
#endif
#endif
  }

  /**
   * \brief Release locks for all keys (for LOCK_ALL strategy
   *        Does nothing if we use LOCK_SINGLE
   */
  inline void unlockAll() {
#if PS_BACKEND_LOCKS
#if PS_BACKEND_LOCK_STRATEGY == LOCK_ALL
    for(size_t i=mu_.size(); i!=0; --i) {
      mu_[i-1].unlock();
    }
#endif
#endif
  }

  /*
    LOCK_SINGLE strategy (PS_BACKEND_LOCK_ALL = 0)
    If we move a parameter (i.e., take it out of the store or put it into the store),
    we lock the entire data structure
  */

  /**
   * \brief Acquire the lock for a single key (for LOCK_SINGLE strategy)
   *        Does nothing if we use LOCK_ALL
   */
  inline void lockSingle(Key key) {
#if PS_BACKEND_LOCK_STRATEGY == LOCK_SINGLE
    mu_[lockForKey(key)].lock();
#endif
  }

  /**
   * \brief Release the lock for a single key (for LOCK_SINGLE strategy)
   *        Does nothing if we use LOCK_ALL
   */
  inline void unlockSingle(Key key) {
#if PS_BACKEND_LOCK_STRATEGY == LOCK_SINGLE
    mu_[lockForKey(key)].unlock();
#endif
  }

  /**
   * \brief Check whether a key is local
   */
  inline bool isLocal(const Key key) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return isLocalUnsafe(key);
  }

  /**
   * \brief Check whether a key is local
   *        Does not acquire the necessary lock
   */
  inline bool isLocalUnsafe(const Key key) {
#if PS_BACKEND < 10
    auto search = store.find(key);
    return search != store.end() && search->second.version != -1;
#else
    return store[key] != nullptr && store[key].version != -1;
#endif
  }

  /**
   * \brief If it is possible to say that a key is non-local
   *        without acquiring the lock, this method does so.
   *        Returns true if the parameter is non-local for sure
   *        Returns false if the parameter (1) is local or (2) it is not
   *        possible to check the status without acquiring a lock
   */
  inline bool isNonLocal_noLock(const Key key) {
#if PS_BACKEND < 10
    return false;
#else
    return store[key] == nullptr;
#endif
  }

  /** Write locality statistics to files */
  void writeStats() {
    // write out locality stats
#if PS_LOCALITY_STATS
    std::string outfile (Postoffice::Get()->get_stats_output_folder() + "/locality_stats.rank." + std::to_string(my_rank) + ".tsv");
    std::ofstream statsfile (outfile, std::ofstream::trunc);
    long total_accesses = 0, total_accesses_local = 0;
    statsfile << "Param\tAccesses\tLocal\n";
    for (uint i=0; i!=num_accesses.size(); ++i) {
      statsfile << i << "\t" << num_accesses[i] << "\t" << num_accesses_local[i] << "\n";
      total_accesses += num_accesses[i];
      total_accesses_local += num_accesses_local[i];
    }
    statsfile.close();
    ADLOG("Wrote locality stats for rank " << my_rank << " to " << outfile << ". Total: " << total_accesses << " accesses, " << total_accesses_local << " local." );
#endif

    // write out key traces
#if PS_TRACE_KEYS
    std::string outfile (Postoffice::Get()->get_stats_output_folder() + "/traces." + std::to_string(my_rank) + ".tsv");
    std::ofstream tracefile (outfile, std::ofstream::trunc);
    std::lock_guard<std::mutex> lk(traces_mu_);
    while (!traces.empty()) {
      auto& trace = traces.front();
      tracefile <<
        std::chrono::duration_cast<std::chrono::milliseconds>(trace.time.time_since_epoch()).count() << "\t" <<
        trace.key << "\t" <<
        my_rank << "\t" <<
        trace.event << "\n";
      traces.pop();
    }
    tracefile.close();
#endif
  }

  /** Returns the length of the value of a specific key */
  inline const size_t get_len(Key key) const {
    if (value_lengths.size() == 1) return value_lengths[0];
    else                           return value_lengths[key];
  }

  /** Returns the sum of the lengths of a list of keys */
  template<typename C>
  inline size_t get_total_len(C& keys) {
    size_t total_len = 0;
    for(Key key : keys)
      total_len += get_len(key);
    return total_len;
  }

  // allow the server to retrieve stats
  unsigned long get_num_pulls_to_replicas() const { return num_pulls_to_replicas; }
  unsigned long get_num_pushs_to_replicas() const { return num_pushs_to_replicas; }
  void reset_replica_stats() { num_pulls_to_replicas=0; num_pushs_to_replicas=0; }

  // get the channel for a key
  const unsigned int get_channel(const Key key) const {
    if (Postoffice::Get()->num_channels() == 1) {
      return 0;
    }

    const std::uint32_t knuth = 2654435769;
    const std::uint32_t y = key;
    return (y * knuth) >> (32 - p_num_channels_);

    // We don't use a naive mod for channel assignment because this causes clashes with key partitioning.
    // The above is a fast(er) implementation of the Knuth multiplication method. This is equivalent to the following:
    // const float A = (sqrt(5)-1)/2;
    // auto slow = std::floor(Postoffice::Get()->num_channels() * (key * A - std::floor(key * A)));
  }

private:

  /** Note down that we have a replica for `key` */
  void rememberReplica(const Key key) {
    auto c = get_channel(key);
    std::lock_guard<std::mutex> lk(replicas_mus[c]);
    replicas[c].insert(key);
  }

  /** Delete the node that we have a replica for `key` */
  void forgetReplica(const Key key) {
    auto c = get_channel(key);
    std::lock_guard<std::mutex> lk(replicas_mus[c]);
    replicas[c].erase(key);
  }

  /** Calculate L2-norm of a parameter vector */
  double diffl2norm(const std::vector<Val>& updated, const std::vector<Val>& old) const {
    double accum = 0;
    for (size_t z=0; z!=updated.size(); ++z) {
      auto diff = updated[z]-old[z];
      accum += (diff)*(diff);
    }
    return sqrt(accum);
  }

  /** Note down one event for one key (write later) */
  inline void trace_key(const Key key, const Event event) {
#if PS_TRACE_KEYS
    if (traced_keys.find(key) != traced_keys.end()) {
      std::lock_guard<std::mutex> lk(traces_mu_);
      traces.emplace(std::chrono::system_clock::now(), key, event);
    }
#endif
  }

  const std::vector<size_t> value_lengths;

#if PS_BACKEND == PS_BACKEND_STD_UNORDERED_LOCKS
  std::unordered_map<Key, Parameter<Val>> store;

#elif PS_BACKEND == PS_BACKEND_VECTOR
  std::vector<std::unique_ptr<Parameter<Val>>> store;

#elif PS_BACKEND == PS_BACKEND_VECTOR_LOCKS
  std::vector<std::unique_ptr<Parameter<Val>>> store;

#endif

#if PS_BACKEND_LOCKS
  // locks
  std::array<std::mutex, PS_BACKEND_NUM_LOCKS> mu_;
#endif

  // a list of all currently held replicas (one separate list per channel)
  std::vector<std::unordered_set<Key>> replicas {};
  std::vector<std::mutex> replicas_mus;

  // the log2 of the number of channels (used for fast channel hashing)
  const int p_num_channels_;

  // replica stats (approximate, as we don't synchronize these counters)
  unsigned long num_pulls_to_replicas = 0;
  unsigned long num_pushs_to_replicas = 0;

  int my_rank; // have a copy of the rank that is available at destruction

  // Contains true for the parameters whose value should be cached
#if PS_LOCALITY_STATS
  std::vector<unsigned long> num_accesses;
  std::vector<unsigned long> num_accesses_local;
#endif

#if PS_TRACE_KEYS
  std::unordered_set<Key> traced_keys {}; // which keys to trace
  std::queue<Trace> traces {}; // stored traces
  std::mutex traces_mu_; // protects `traces`
#endif
};

}  // namespace ps
#endif  // PS_COLOC_SERVER_HANDLE_
