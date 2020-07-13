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
      Parameter(size_t vpk): val(vpk) {}
      // Constructor: normal feature, initial value given
      Parameter(Val* init_val, size_t vpk): val(init_val, vpk) {}
      Parameter() {}

      // the parameter value
      std::valarray<Val> val;

      // for value caches: cached parameter updates
      std::valarray<Val> cached_updates;

      // indicates whether this feature is cached
      bool cache = false;

      // indicates whether there is a cached value for this feature right now
      bool have_cached_value = false;

      // indicates whether this parameter was updated since the last cache sync
      bool updated = false;
    };

template <typename Val>
struct DefaultColoServerHandle {
public:

DefaultColoServerHandle(long num_keys, size_t v): transfers{num_keys}, vpk{v} {
  long store_max = Postoffice::Get()->GetServerKeyRanges().back().end();
  ADLOG("Creating handle for " << num_keys << " (max " << store_max << ") keys with vpk " << vpk);

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
    store = new Val[store_max*vpk];
    std::fill_n(store, store_max, 0);

#elif PS_BACKEND == PS_BACKEND_ARRAY_ATOMIC
    std::cout << "Handle data structure: std::atomic<Val>[]" << std::endl;
    store = new std::atomic<Val>[store_max];
    std::fill_n(store, store_max, 0);

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
    // clean stats directory
    if (myRank == 0 && system("exec rm -r stats/locality_stats.rank.*.tsv")==0)
      ADLOG("Cleaned locality statistics");
#endif

    // Insert default values for local keys
    const Range& myRange = Postoffice::Get()->GetServerKeyRanges()[Postoffice::Get()->my_rank()];
    lockAll(); // does nothing if we use LOCK-SINGLE
    for(Key i=myRange.begin(); i!=myRange.end(); ++i) {
      lockSingle(i); // does nothing if we use LOCK-ALL
      insertKeyUnsafe(i);
      unlockSingle(i);
    }
    unlockAll();
  }

  ~DefaultColoServerHandle(){
#if  PS_BACKEND == PS_BACKEND_ARRAY_ATOMIC
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
      store[key] = Parameter<Val>(vpk); // use default value for Val (typically 0)
    } else {
      store[key] = Parameter<Val>(val, vpk);
    }
# else
    if (val == 0) { // default value
      store[key] = make_unique<Parameter<Val>>(vpk);
    } else {
      store[key] = make_unique<Parameter<Val>>(val, vpk);
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
    // erase key from local store
#if PS_BACKEND < 10 // map
    store.erase(key);
#elif PS_BACKEND == PS_BACKEND_VECTOR_LOCKS || PS_BACKEND == PS_BACKEND_VECTOR
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
#if PS_LOCALITY_STATS
    num_accesses[key]++;
#endif
    // attempt to push the value for this key into the local store
#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end()) {
      return false;
    } else {
      mergeValue(search->second.val, val);

      // if this feature is cached, we additionally note the updates separately,
      // so that we can send delta updates to the server later
      if (search->second.cache) {
        mergeValue(search->second.cached_updates, val);
        search->second.updated = true;
      }
#if PS_LOCALITY_STATS
      num_accesses_local[key]++;
#endif
      return true;
    }
#else
    if (store[key] == nullptr) {
      return false;
    } else {
      auto& param = *store[key];
      mergeValue(param.val, val);

      if (param.cache) {
        mergeValue(param.cached_updates, val);
        param.updated = true;
      }
#if PS_LOCALITY_STATS
      num_accesses_local[key]++;
#endif
      return true;
    }
#endif
  }

  /**
   * \brief Merge a value `merge` into an existing value `target`
   */
  template<typename V1, typename V2>
    inline void mergeValue(V1& target, V2& merge) {
    for(uint i=0; i!=vpk; ++i) {
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
  inline bool attemptLocalPull(const Key key, Val* val) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
    return attemptLocalPullUnsafe(key, val);
  }



  /**
     \brief Process a local or remote pull request
   */
  inline ps::Status processPull(const Key key, Val* val, int ts, Customer* customer, bool localRequest=true, std::shared_ptr<QueuedMessage<Val>> queued_msg = {}) {
#if PS_BACKEND_LOCKS
    std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif

    if ((Postoffice::Get()->shared_memory_access() || !localRequest) && attemptLocalPullUnsafe(key, val)) {
      return ps::Status::LOCAL;
    } else if (transfers.isInTransferUnsafe(key)) {
      if (localRequest)
        transfers.queueLocalRequestUnsafe(key, ts, customer);
      else
        transfers.queueRemoteRequestUnsafe(key, queued_msg);
      transfers.addPullToQueueUnsafe(key, val);
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
        transfers.addPushToQueueUnsafe(key, val, vpk);
      } else {
        transfers.queueRemoteRequestUnsafe(key, queued_msg);
        transfers.addRemotePushToQueueUnsafe(key, val, vpk, data_ptr);
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
        transfers.startTransferUnsafe(key, vpk);
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
  inline bool attemptLocalPullUnsafe(const Key key, Val* val) {
#if PS_LOCALITY_STATS
    num_accesses[key]++;
#endif
    // attempt to pull the value of this key from the local store
#if PS_BACKEND < 10 // map
    auto search = store.find(key);
    if (search == store.end()) {
      return false; // this key is not local
    } else {
      if (search->second.cache && !search->second.have_cached_value) {
        // when using value caches, we might have an entry in the backend data structure
        // although we don't currently have a value for this entry. in this case, return false
        return false;
      } else {
        std::copy_n(begin(search->second.val), vpk, val);
#if PS_LOCALITY_STATS
        num_accesses_local[key]++;
#endif
        return true;
      }
    }
#else
    if (store[key] == nullptr) {
      return false; // this key is not local
    } else {
      auto& param = (*store[key]);
      if (param.cache && !param.have_cached_value) {
        // when using value caches, we might have an entry in the backend data structure
        // although we don't currently have a value for this entry. in this case, return false
        return false;
      } else {
        std::copy_n(begin(param.val), vpk, val);
#if PS_LOCALITY_STATS
        num_accesses_local[key]++;
#endif
        return true;
      }
    }
#endif
  }

  /**
   * \brief Initialize value caches
   */
  void initValueCaches(std::vector<bool>* parameters_to_cache) {
    cached_parameters = parameters_to_cache;
    lockAll();
    // insert entries for cached parameters
    for (unsigned long i=0; i!=cached_parameters->size(); ++i) {
      if ((*cached_parameters)[i]) {
        lockSingle(i);
#if PS_BACKEND < 10 // map
        auto search = store.find(i);
        if (search == store.end()) {
#else
        if(store[i] == nullptr) {
#endif
          insertKeyUnsafe(i);

#if PS_BACKEND < 10 // map
          auto& param = store[i];
#else
          auto& param = (*store[i]);
#endif

          param.cache = true;
          param.cached_updates.resize(vpk);
        }
        unlockSingle(i);
      }
    }
    unlockAll();
  }

  /**
   * \brief Update the value cache for one parameter (e.g., when a remote pull arrives)
   */
  inline void updateValueCache(const Key key, Val* val) {
    // update the value cache if this is a cached parameter
    if ((*cached_parameters)[key]) {
#if PS_BACKEND_LOCKS
      std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
#if PS_BACKEND < 10 // map
      auto search = store.find(key);
      CHECK(search != store.end()) << "Cannot find entry for cached parameter " << key;
      auto& param = search->second;
# else
      CHECK(store[key] != nullptr) << "Didn't find";
      auto& param = (*store[key]);
#endif

      // update cache only if this parameter is really cached (and not if it is a local parameter)
      if (param.cache) {
        std::copy_n(begin(param.val), vpk, val);

        // apply any updates
        if (param.updated) {
          mergeValue(param.val, param.cached_updates);
        }
        param.have_cached_value = true;
      }
    }
  }

  /**
   * \brief Clear all value caches and collect cached updates
   */
  unsigned long clearValueCachesAndCollectUpdates(std::vector<KVPairs<Val>>& kvs, ps::Addressbook& addressbook, bool collectUpdates = true) {
    CHECK (cached_parameters != nullptr) << "Caching is not initialized. Thus, cannot clear caches.";
    auto total_params = cached_parameters->size();
    unsigned long num_keys = 0;
    for (unsigned long key=0; key!=total_params; ++key) {
      if ((*cached_parameters)[key]) {
#if PS_BACKEND_LOCKS
        std::lock_guard<std::mutex> lk(mu_[lockForKey(key)]);
#endif
#if PS_BACKEND < 10 // map
        auto& param = store[key];
#else
        auto& param = (*store[key]);
#endif
        if (param.cache) {
          param.have_cached_value = false;

          if (collectUpdates && param.updated) {
            ++num_keys;

            // copy out updates
            auto destination = addressbook.getManager(key);
            kvs[destination].keys.push_back(key);
            std::copy_n(std::begin(param.cached_updates), vpk, std::back_inserter(kvs[destination].vals));

            // clear updates
            std::fill_n(std::begin(param.cached_updates), vpk, 0);
            param.updated = false;
          }
        }
      }
    }
    return num_keys;
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
    return store.find(key) != store.end();
#else
    return store[key] != nullptr;
#endif
  }

  const size_t get_vpk() {
    return vpk;
  }

  /** Write locality statistics to files */
  void writeStats() {
#if PS_LOCALITY_STATS
    ofstream statsfile ("stats/locality_stats.rank." + std::to_string(myRank) + ".tsv", ofstream::trunc);
    statsfile << "Param\tAccesses\tLocal\n";
    for (uint i=0; i!=num_accesses.size(); ++i) {
      statsfile << i << "\t" << num_accesses[i] << "\t" << num_accesses_local[i] << "\n";
    }
    statsfile.close();
#endif
  }


  ColoServerTransfers<Val> transfers; // TODO: make private

private:
  size_t vpk;

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
  std::array<std::mutex, PS_BACKEND_NUM_LOCKS> mu_;
#endif

  // Contains true for the parameters whose value should be cached
  std::vector<bool>* cached_parameters = nullptr;
#if PS_LOCALITY_STATS
  std::vector<unsigned long> num_accesses;
  std::vector<unsigned long> num_accesses_local;
  int myRank;
#endif
};

}  // namespace ps
#endif  // PS_COLOC_SERVER_HANDLE_
