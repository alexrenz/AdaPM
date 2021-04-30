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
#include "ps/coloc_kv_transfers.h"
#include "../apps/utils.h"
#include <valarray>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>

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

   The desired backend data structure can be specified with the compilation flag PS_BACKEND in
   the external variable LAPSE_EXTERNAL_LDFLAGS. E.g, the following compiles the knowledge graph
   embeddings app with an unordered map:
   export LAPSE_EXTERNAL_LDFLAGS="-DPS_BACKEND=1"; make apps/knowledge_graph_embeddings

*/


// PS back-end data structure variants
#define PS_BACKEND_STD_UNORDERED_LOCKS 1
#define PS_BACKEND_GOOGLE_DENSE 2
#define PS_BACKEND_TBB_CONCURRENT_UNORDERED 4
#define PS_BACKEND_TBB_CONCURRENT_HASH_MAP 5
#define PS_BACKEND_VECTOR 10
#define PS_BACKEND_VECTOR_LOCKS 14
#define PS_BACKEND_ARRAY 11
#define PS_BACKEND_ARRAY_ATOMIC 13

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
#define PS_BACKEND_NUM_LOCKS 1000
#endif

// if we have a vector backend structure, we use LOCK_SINGLE by default, otherwise we use LOCK_ALL
#ifndef PS_BACKEND_LOCK_STRATEGY
#define PS_BACKEND_LOCK_STRATEGY PS_BACKEND != PS_BACKEND_VECTOR_LOCKS
#endif

// backend locality statistics
#ifndef PS_LOCALITY_STATS
#define PS_LOCALITY_STATS 0
#endif
namespace ps {

  // relocation statistics
#ifdef PS_RELOC_STATS
  extern std::atomic<long> STATS_num_localizes;
  extern std::atomic<long> STATS_num_localizes_local;
  extern std::atomic<long> STATS_num_localizes_queue;
#endif

  /* A parameter in the local parameter server */
  template <typename Val>
    struct Parameter {
      // Constructor: normal feature, no inital value
      Parameter(size_t len): val(len) {}
      // Constructor: normal feature, initial value given
      Parameter(Val* init_val, size_t len): val(init_val, len) {}
      Parameter() {}

      // the parameter value
      std::valarray<Val> val;

      // for value caches: cached parameter updates
      std::valarray<Val> cached_updates;

      // indicates whether this feature is cached
      bool cache = false;

      // indicates whether this parameter was updated since the last cache sync
      bool updated = false;

      // data for keeping around values of unused parameters
      bool keptAround = false; // is the available value a kept around value?
      chrono::time_point<std::chrono::steady_clock> removeTime; // time of removal
      long numUses = 0; // number of times this parameter was accessed while at this node
    };

template <typename Val>
struct DefaultColoServerHandle {
public:

  // Uniform parameter lengths: each parameter value has the same length
  DefaultColoServerHandle(size_t num_keys, size_t uniform_len): DefaultColoServerHandle(num_keys, std::vector<size_t> {uniform_len}) { }

  // Non-uniform parameter lengths: each parameter can have its own value length
  DefaultColoServerHandle(size_t num_keys, const std::vector<size_t>& value_lengths):
    transfers {num_keys}, value_lengths {value_lengths},
    keepAroundUnused{ColoKVServer<Val,DefaultColoServerHandle>::keepAroundUnused != 0},
    maxKeepAround{std::chrono::microseconds(ColoKVServer<Val,DefaultColoServerHandle>::keepAroundUnused)} {

    long store_max = Postoffice::Get()->GetServerKeyRanges().back().end();
    ADLOG("Creating handle for " << num_keys << " (max " << store_max << ") keys with " <<
          (value_lengths.size() == 1 ? "uniform" : "non-uniform") << " lengths " <<
          " (keep around: " << std::chrono::duration_cast<std::chrono::microseconds>(maxKeepAround).count() << " us)");

#if PS_BACKEND == PS_BACKEND_STD_UNORDERED_LOCKS
    std::cout << "Handle data structure: std::unordered_map with " << mu_.size() << " locks" << std::endl;
    store.reserve(store_max);

#elif PS_BACKEND == PS_BACKEND_GOOGLE_DENSE
    std::cout << "Handle data structure: Google dense" << std::endl;
    store.set_empty_key(-1);

#elif PS_BACKEND == PS_BACKEND_TBB_CONCURRENT_UNORDERED
    std::cout << "Handle data structure: TBB concurrent unordered" << std::endl;

#elif PS_BACKEND == PS_BACKEND_TBB_CONCURRENT_HASH_MAP
    std::cout << "Handle data structure: TBB concurrent hash map" << std::endl;

#elif PS_BACKEND == PS_BACKEND_VECTOR
    std::cout << "Handle data structure: vector<unique_ptr<Parameter>> (no locks)" << std::endl;
    store.resize(store_max);

#elif PS_BACKEND == PS_BACKEND_VECTOR_LOCKS
    std::cout << "Handle data structure: vector<unique_ptr<Parameter>> with " << mu_.size() << " locks" << std::endl;
    store.resize(store_max);

#elif PS_BACKEND == PS_BACKEND_ARRAY
    std::cout << "Handle data structure: Val[]" << std::endl;
    CHECK(false) << "The chosen backend data structure is not implemented";
    // store = new Val[length_of_all_keys];
    // std::fill_n(store, store_max, 0);

#elif PS_BACKEND == PS_BACKEND_ARRAY_ATOMIC
    std::cout << "Handle data structure: std::atomic<Val>[]" << std::endl;
    CHECK(false) << "The chosen backend data structure is not implemented";
    // store = new std::atomic<Val>[store_max];
    // std::fill_n(store, store_max, 0);

#else
    std::cout << "Handle data structure: std::unordered" << std::endl;

#endif

#if PS_BACKEND_LOCKS
#if PS_BACKEND_LOCK_STRATEGY == LOCK_ALL
    std::cout << "Locking strategy: LOCK_ALL" << std::endl;
#elif PS_BACKEND_LOCK_STRATEGY == LOCK_SINGLE
    std::cout << "Locking strategy: LOCK_SINGLE" << std::endl;
#endif
#endif

#if PS_LOCALITY_STATS
    ADLOG("Capture locality statistics for " << store_max << " keys");
    myRank = Postoffice::Get()->my_rank();
    num_accesses.resize(store_max, 0);
    num_accesses_local.resize(store_max, 0);
    num_accesses_in_transfer.resize(store_max, 0);
    num_kept_around.resize(store_max, 0);
    num_kept_around_used.resize(store_max, 0);
    // clean stats directory
    if (myRank == 0 && std::system("mkdir -p stats && rm -r stats/locality_stats*.rank.*.tsv")==0)
      ADLOG("Cleaned locality statistics");
#endif

    // Insert default values for local keys
    const Range& myRange = Postoffice::Get()->GetServerKeyRanges()[Postoffice::Get()->my_rank()];
    lockAll(); // does nothing if we use LOCK-SINGLE
    for(Key i=myRange.begin(); i!=myRange.end(); ++i) {
      lockSingle(i); // does nothing if we use LOCK-ALL
      if(i < num_keys) {
        insertKeyUnsafe(i);
#if PS_BACKEND < 10 // map
        store[i].numUses = 1;
# else
        store[i]->numUses = 1; // disable keep around for initial allocation
#endif
      }
      unlockSingle(i);
    }
    unlockAll();
  }

  ~DefaultColoServerHandle(){
#if PS_BACKEND == PS_BACKEND_ARRAY_ATOMIC
    delete[] store;
#endif
  }

#if PS_BACKEND_LOCKS
  inline size_t lockForKey(Key key) {
    return key % mu_.size();
  }
#endif

  /**
   * \brief Inserts a key into the local data structure
   */
  void insertKey(Key key, Val* val=0) {
    lockAll();
    lockSingle(key);
    insertKeyUnsafe(key, val);
    unlockSingle(key);
    unlockAll();
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
      store[key] = make_unique<Parameter<Val>>(get_len(key));
    } else {
      store[key] = make_unique<Parameter<Val>>(val, get_len(key));
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
   *        If the deleted key has not been used while at this node (and keep around is enabled),
   *        we keep its value around for a short period of time (to serve pulls).
   *
   *        Warning: this method is not thread safe.
   */
  inline void removeKeyUnsafe(Key key) {
    // special case: keep a cached value around if the key was not used
#if PS_BACKEND < 10 // map
    if (keepAroundUnused && store[key].numUses == 1) {
# else
    if (keepAroundUnused && store[key]->numUses == 1) {
#endif
      // (`== 1` assumes that a relocation calls `attemptLocalPull` once before `removeKey`
#if PS_BACKEND < 10 // map
      auto search = store.find(key);
      if (search != store.end()) {
        search->second.keptAround = true;
        search->second.removeTime = std::chrono::steady_clock::now();
#if PS_LOCALITY_STATS
        num_kept_around[key]++;
#endif
      }
#else
      if (store[key] != nullptr) {
        store[key]->keptAround = true;
        store[key]->removeTime = std::chrono::steady_clock::now();
#if PS_LOCALITY_STATS
        num_kept_around[key]++;
#endif
      }
#endif
    } else { // normal case: delete the key from the store
      dropKeyUnsafe(key);
    }
  }

  /**
   * \brief Attempt to push a the value for a key into this parameter server
   *
   * @param key the key
   * @param val a pointer to the corresponding values
   * @return true if the key is local (and therefore, was pushed), false otherwise
   */
  inline bool attemptLocalPush(const Key key, const Val* val) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return attemptLocalPushUnsafe(key, val);
  }

  /**
   * \brief Attempt to push a the value for a key into this parameter server
   *
   *        Warning: this method is not thread safe.
   */
  inline bool attemptLocalPushUnsafe(const Key key, const Val* val) {
    // attempt to push the value for this key into the local store
#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end() || search->second.keptAround) {
      return false;
    } else {
      mergeValue(search->second.val, val, get_len(key));

      // if this parameter is replicated, we additionally note the updates separately,
      // so that we can send delta updates to the server later
      if (search->second.cache) {
        mergeValue(search->second.cached_updates, val, get_len(key));
        search->second.updated = true;
        ++num_pushs_to_replicas;
      }
      search->second.numUses++;
      return true;
    }
#else
    if (store[key] == nullptr || store[key]->keptAround) {
      return false;
    } else {
      auto& param = *store[key];
      mergeValue(param.val, val, get_len(key));

      if (param.cache) {
        mergeValue(param.cached_updates, val, get_len(key));
        param.updated = true;
        ++num_pushs_to_replicas;
      }
      ++param.numUses;
      return true;
    }
#endif
  }

  /**
   * \brief Merge a value `merge` into an existing value `target`
   */
  template<typename V1, typename V2>
    inline void mergeValue(V1& target, V2& merge, const size_t len) {
    for(uint i = 0; i != len; ++i) {
      target[i] += merge[i];
    }
  }

  /**
   * \brief Attempt to pull the value of a key from this parameter server
   *
   * @param key the key
   * @param vals the array to put the value into
   * @return true if the key is local (and therefore, the current value is in vals), false otherwise
   */
  inline bool attemptLocalPull(const Key key, Val* val, const bool stats=true, const bool regularWorkerCall=false) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return attemptLocalPullUnsafe(key, val, stats, regularWorkerCall);
  }



  /**
     \brief Process a local or remote pull request
   */
  inline ps::Status processPull(const Key key, Val* val, int ts, Customer* customer, bool localRequest=true, std::shared_ptr<QueuedMessage<Val>> queued_msg = {}) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

    if ((Postoffice::Get()->shared_memory_access() || !localRequest) && attemptLocalPullUnsafe(key, val, false)) {
      return ps::Status::LOCAL;
    } else if (transfers.isInTransferUnsafe(key)) {
      if (localRequest)
        transfers.queueLocalRequestUnsafe(key, ts, customer);
      else
        transfers.queueRemoteRequestUnsafe(key, queued_msg);
      transfers.addPullToQueueUnsafe(key, val);
#if PS_LOCALITY_STATS
    num_accesses_in_transfer[key]++;
#endif
      return ps::Status::IN_TRANSFER;
    } else {
      return ps::Status::REMOTE;
    }
  }


  /**
     \brief Process a local or remote push request
  */
  inline ps::Status processPush(const Key key, Val* val, const int ts, Customer* customer, bool localRequest=true,
                                std::shared_ptr<QueuedMessage<Val>> queued_msg = {},
                                std::shared_ptr<KVPairs<Val>> data_ptr = {}) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

    if ((Postoffice::Get()->shared_memory_access() || !localRequest) && attemptLocalPushUnsafe(key, val)) {
      return ps::Status::LOCAL;
    } else if (transfers.isInTransferUnsafe(key)) {
      if (localRequest) {
        transfers.queueLocalRequestUnsafe(key, ts, customer);
        transfers.addLocalPushToQueueUnsafe(key, val, get_len(key));
      } else {
        transfers.queueRemoteRequestUnsafe(key, queued_msg);
        transfers.addRemotePushToQueueUnsafe(key, val, data_ptr);
      }

      return ps::Status::IN_TRANSFER;
    } else {
      return ps::Status::REMOTE;
    }
  }

  /**
     \brief Process a local localize request
   */
  inline ps::Status processLocalize(const Key key, const int ts, Customer* customer) {

    // special case: don't localize cached parameters
    if (cached_parameters != nullptr && (*cached_parameters)[key]) {
      return ps::Status::LOCAL;
    } else {

#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

#if PS_RELOC_STATS
      ++STATS_num_localizes;
#endif

      if (isLocalUnsafe(key)) {
        // skip the localization: key is already local
#if PS_RELOC_STATS
        ++STATS_num_localizes_local;
#endif
        return ps::Status::LOCAL;
      } else if (transfers.isInTransferUnsafe(key)) {
        // wait for a previously started localize
        transfers.queueLocalRequestUnsafe(key, ts, customer);
#if PS_RELOC_STATS
        ++STATS_num_localizes_queue;
#endif
        return ps::Status::IN_TRANSFER;
      } else {
        // start a new localize
        transfers.startTransferUnsafe(key);
        transfers.queueLocalRequestUnsafe(key, ts, customer);
        return ps::Status::REMOTE;
      }
    }
  }

  /**
     \brief Determine whether a key is in transfer to the local PS right now
   */
  bool isInTransfer(Key key) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

    return transfers.isInTransferUnsafe(key);
  }

  /**
     \brief Store a subsequent localize (to another PS) for an ongoing localize (to the local PS)
  */
  bool noteSubsequentLocalize(Key key, std::shared_ptr<QueuedMessage<Val>>& queued_msg) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

    return transfers.noteSubsequentLocalizeUnsafe(key, queued_msg);
  }

  /**
   * \brief Attempt to pull the value of a key from this parameter server
   *
   *        Warning: this method is not thread safe.
   */
  inline bool attemptLocalPullUnsafe(const Key key, Val* val, const bool stats=true, const bool regularWorkerCall=false) {
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

      // key is local

      // special case: if this is a "kept around" value, check whether it is current enough
      if (param.keptAround) {
        if (!regularWorkerCall) { // use kept values only for `Pull()` calls (and not for system functions or `PullIfLocal`)
          return false;
        }
        if (std::chrono::steady_clock::now() - param.removeTime > maxKeepAround) {
          // too old: delete the kept value
          dropKeyUnsafe(key);
          return false;
        }
      }

      // the actual pull:copy the stored value to the passed location
      std::copy_n(begin(param.val), get_len(key), val);
      ++param.numUses;

      // stats
      if (param.cache) {
        ++num_pulls_to_replicas;
      }

      // stats
#if PS_LOCALITY_STATS
      if (stats) num_accesses_local[key]++;
#endif

      // delete the value if this was a "kept around value"
      if (param.keptAround) {
#if PS_LOCALITY_STATS
        num_kept_around_used[key]++;
#endif
        dropKeyUnsafe(key);
      }
      return true;
    }
  }

  /**
   * \brief Initialize replication of specified parameters
   */
  void initializeReplication(SArray<Key>& replicated) {
    lockAll();
    // insert entries for cached parameters
    // for (unsigned long i=0; i!=cached_parameters->size(); ++i) {
    for (Key key : replicated) {
      lockSingle(key);
#if PS_BACKEND < 10 // map
      auto search = store.find(key);
      if (search == store.end())
#else
      if(store[key] == nullptr)
#endif
      {
        insertKeyUnsafe(key);
      }

#if PS_BACKEND < 10 // map
      auto& param = store[key];
#else
      auto& param = (*store[key]);
#endif

      param.cache = true;
      param.cached_updates.resize(get_len(key));
      unlockSingle(key);
    }
    unlockAll();
  }

  /**
   * \brief Read locally accumulated updates of replicated parameters.
   *
   *        Synchronizes selectively, depending on `threshold`:
   *        - threshold=-1: synchronize all parameters (including ones with no updates)
   *        - threshold=0: synchronize parameters with non-zero updates
   *        - threshold>0: synchronize parameters where norm(updates)>threshold
   *
   */
  void readReplicas(const SArray<Key>& keys, KVPairs<Val>& updates, const double threshold=-1) {

    assert(updates.keys.size() == 0);
    assert(updates.vals.size() == 0);

    for (unsigned long i=0; i!=keys.size(); ++i) {
      Key key = keys[i];
#if PS_BACKEND_LOCKS
      std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
#if PS_BACKEND < 10 // map
      auto& param = store[key];
#else
      auto& param = (*store[key]);
#endif
      assert(param.cache);

      // extract update
      if ((threshold == -1) ||
          (threshold == 0 && param.updated) ||
          (threshold > 0 && param.updated && l2norm(param.cached_updates) >= threshold)) {

        updates.keys.push_back(key);

        // extract accumulated updates
        std::copy_n(std::begin(param.cached_updates), get_len(key), std::back_inserter(updates.vals));

        // clear accumulated updates (set to 0)
        std::fill_n(std::begin(param.cached_updates), get_len(key), 0);
        param.updated = false;
      }
    }
  }


  /**
   * \brief Write new parameter value to a replicated parameter
   *
   *        Adds updates that were accumulated since the synchronization mechanism
   *        last read the replica.
   */
  void writeReplica(const Key key, const Val* state) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
#if PS_BACKEND < 10 // map
    auto& param = store[key];
#else
    auto& param = (*store[key]);
#endif
    assert(param.cache);

    // update the replica (adds the local updates that occurred since the sync last read the value)
    for (size_t j=0; j!=get_len(key); ++j) {
      param.val[j] = state[j] + param.cached_updates[j];
    }
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
    return search != store.end() && !search->second.keptAround;
#else
    return store[key] != nullptr && !store[key]->keptAround;
#endif
  }

  /**
   * \brief If it is possible to say that a key is non-local
   *        without acquiring the lock, this method does so.
   *        Returns true if the parameter is non-local for sure
   *        Returns false if the parameter (1) is local or (2) it is not
   *        possible to check the status without acquiring a lock
   */
  inline bool nolockNonLocalCheck(const Key key) {
#if PS_BACKEND < 10
    return false;
#else
    return store[key] == nullptr;
#endif
  }

  /** Write locality statistics to files */
  void writeStats() {
#if PS_LOCALITY_STATS
    std::string outfile ("stats/locality_stats.rank." + std::to_string(myRank) + ".tsv");
    ofstream statsfile (outfile, ofstream::trunc);
    long total_accesses = 0, total_accesses_local = 0, total_accesses_in_transfer = 0, total_kept_around = 0, total_kept_around_used = 0;
    statsfile << "Param\tAccesses\tLocal\tInTransfer\tKeptAround\tKeptAroundUsed\n";
    for (uint i=0; i!=num_accesses.size(); ++i) {
      statsfile << i << "\t" << num_accesses[i] << "\t" << num_accesses_local[i] << "\t" << num_accesses_in_transfer[i] << "\t" << num_kept_around[i] << "\t" << num_kept_around_used[i] << "\n";
      total_accesses += num_accesses[i];
      total_accesses_local += num_accesses_local[i];
      total_accesses_in_transfer += num_accesses_in_transfer[i];
      total_kept_around += num_kept_around[i];
      total_kept_around_used += num_kept_around_used[i];
    }
    statsfile.close();
    ADLOG("Wrote locality stats for rank " << myRank << " to " << outfile << ". Total: " << total_accesses << " accesses, " << total_accesses_local << " local, " << total_accesses_in_transfer << " in transfer. \n" << 100.0 * total_kept_around_used / total_kept_around << "% of " << total_kept_around << " kept around used" );
#endif
  }

  /** Returns the length of the value of a specific key */
  inline const size_t get_len(Key key) {
    if (value_lengths.size() == 1) return value_lengths[0];
    else                           return value_lengths[key];
  }

  /** Returns the sum of the lengths of a list of keys */
  inline size_t get_total_len(SArray<Key>& keys) {
    size_t total_len = 0;
    for(Key key : keys)
      total_len += get_len(key);
    return total_len;
  }

  // allow the server to retrieve stats
  unsigned long get_num_pulls_to_replicas() const { return num_pulls_to_replicas; }
  unsigned long get_num_pushs_to_replicas() const { return num_pushs_to_replicas; }
  void reset_replica_stats() { num_pulls_to_replicas=0; num_pushs_to_replicas=0; }

  ColoServerTransfers<Val> transfers; // TODO: make private
private:

  /** Actually delete a key from the local store */
  inline void dropKeyUnsafe(Key key) {
#if PS_BACKEND < 10 // map
    store.erase(key);
#else
    store[key].reset();
#endif
  }

  /** Calculate L2-norm of a parameter vector */
  double l2norm(const std::valarray<Val>& vals) const {
    double accum = 0;
    for (Val val : vals) {
      accum += val*val;
    }
    return sqrt(accum);
  }

  const std::vector<size_t> value_lengths;

#if PS_BACKEND == PS_BACKEND_STD_UNORDERED_LOCKS
  std::unordered_map<Key, Parameter<Val>> store;

#elif PS_BACKEND == PS_BACKEND_GOOGLE_DENSE
  google::dense_hash_map<Key,Val> store;

#elif PS_BACKEND == PS_BACKEND_TBB_CONCURRENT_UNORDERED
  tbb::concurrent_unordered_map<Key, std::valarray<Val>> store; // doesn't support insert/remove

#elif PS_BACKEND == PS_BACKEND_TBB_CONCURRENT_HASH_MAP
  tbb::concurrent_hash_map<Key, std::valarray<Val>> store;

#elif PS_BACKEND == PS_BACKEND_VECTOR
  std::vector<std::unique_ptr<Parameter<Val>>> store;

#elif PS_BACKEND == PS_BACKEND_VECTOR_LOCKS
  std::vector<std::unique_ptr<Parameter<Val>>> store;

#elif PS_BACKEND == PS_BACKEND_ARRAY
  Val* store;

#elif PS_BACKEND == PS_BACKEND_ARRAY_ATOMIC
  std::atomic<Val>* store;

#else
  std::unordered_map<Key, Parameter<Val>> store;

#endif

#if PS_BACKEND_LOCKS
  // locks
  std::array<std::mutex, PS_BACKEND_NUM_LOCKS> mu_;
#endif

  // settings for keeping around unused parameters
  const bool keepAroundUnused;
  const std::chrono::steady_clock::duration maxKeepAround;

  // replica stats (approximate, as we don't synchronize these counters)
  unsigned long num_pulls_to_replicas = 0;
  unsigned long num_pushs_to_replicas = 0;

  // Contains true for the parameters whose value should be cached
  std::vector<bool>* cached_parameters = nullptr;
#if PS_LOCALITY_STATS
  std::vector<unsigned long> num_accesses;
  std::vector<unsigned long> num_accesses_local;
  std::vector<unsigned long> num_accesses_in_transfer;
  std::vector<unsigned long> num_kept_around;
  std::vector<unsigned long> num_kept_around_used;
  int myRank;
#endif
};

}  // namespace ps
#endif  // PS_COLOC_SERVER_HANDLE_
