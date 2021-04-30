/**
 *  Sampling support in a parameter server
 *
 *  Implements different approaches for supporting sampling, i.e., pulling any
 *  of a specified group of keys, according to a specified distribution.
 *
 *  Currently implemented strategies:
 *   - Naive
 *   - Pre-localization
 *   - Pool
 *   - Only local
 *
 *  Copyright (c) 2020 Alexander Renz-Wieland
 */

#ifndef PS_SAMPLING_H_
#define PS_SAMPLING_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <unordered_map>
#include <ps/internal/postoffice.h>
#include <math.h>
#include <random>
#include <iostream>
#include <sstream>
#include <queue>

namespace ps {

  // Strategies for sampling support
  enum class SamplingSupportType {Naive, Preloc, Pool, OnlyLocal};

  std::ostream& operator<<(std::ostream &o, const SamplingSupportType& smt) {
    switch(smt) {
    case SamplingSupportType::Naive: return o << "naive";
    case SamplingSupportType::Preloc: return o << "preloc";
    case SamplingSupportType::Pool: return o << "pool";
    case SamplingSupportType::OnlyLocal: return o << "onlylocal";
    default: return o << "unkown";
    }
  }
  std::istream& operator>>(std::istream& in, SamplingSupportType& smt) {
    std::string token; in >> token;
    if (token == "naive") smt = SamplingSupportType::Naive;
    else if (token == "preloc") smt = SamplingSupportType::Preloc;
    else if (token == "pool") smt = SamplingSupportType::Pool;
    else if (token == "onlylocal") smt = SamplingSupportType::OnlyLocal;
    else { CHECK(false) << "Fatal! Unkown replica synchronization method " << token; }
    return in;
  }


  // per-sample information that we store between the prep and the pull call
  struct PrelocSample {
    std::vector<Key> keys;
    size_t used=0;
  };

  // per-sample information for pool sampling support
  struct PoolSample {
    std::vector<Key> keys;
    size_t used=0;
    size_t K=0;
  };

  /**
   * \brief Base class for sampling support
   */
  template <typename Val, typename Worker>
    class SamplingSupport {
  protected:
    std::default_random_engine generator;

    // pointer to (application-provided) function to sample 1 key
    Key (*const sample_key_)();

    // counters for creating unique sample IDs (one per customer)
    std::vector<size_t> id_counters;
    std::vector<Worker*> workers;

    // stats about number of non-deterministic accesses
#if PS_LOCALITY_STATS
    std::vector<unsigned long> num_accesses;
    int myRank;
#endif

    // count key accesses
    inline void count_accesses(std::vector<Key>& keys) {
#if PS_LOCALITY_STATS
      for (Key key : keys) {
        num_accesses[key]++;
      }
#endif
    }

  public:
    SamplingSupport(Key (*const sample_key)()):
      sample_key_(sample_key), id_counters(Postoffice::Get()->num_worker_threads(), 1),
      workers(Postoffice::Get()->num_worker_threads(), nullptr) {
#if PS_LOCALITY_STATS
      ADLOG("Capture number of non-deterministic accesses per key");
      myRank = Postoffice::Get()->my_rank();
      num_accesses.resize(Postoffice::Get()->max_key(), 0);
#endif
    }

    virtual ~SamplingSupport() {};

    // hook to terminate any activities
    virtual void terminate() {
#if PS_LOCALITY_STATS
      // write number of non-deterministic accesses to file
      std::string outfile ("stats/locality_stats.sampling.rank." + std::to_string(myRank) + ".tsv");
      ofstream statsfile (outfile, ofstream::trunc);
      long total_accesses = 0;
      statsfile << "Param\tAccesses\n";
      for (uint i=0; i!=num_accesses.size(); ++i) {
        statsfile << i << "\t" << num_accesses[i] << "\n";
        total_accesses += num_accesses[i];
      }
      statsfile.close();
      ADLOG("Wrote non-deterministic access stats for rank " << myRank << " to " << outfile << ". Total: " << total_accesses << " accesses");
#endif
    }

    virtual void registerWorker(int customer_id, Worker* worker) {
      workers[customer_id-1] = worker;
    }

    // prepare a sample of `K` keys; returns a sample ID
    virtual SampleID prepare_sample(size_t K, int customer_id) = 0;

    // pull N (=keys.size()) keys from the prepared sample with ID `id`; returns an operation timestamp
    virtual int pull_sample(SampleID sample_id, std::vector<Key>& keys, std::vector<Val>& vals, const int customer_id) = 0;

    // declare a sample as finished (so that any related data can be deleted)
    virtual void finish_sample(SampleID sample_id, int customer_id) {};

    // program options
    static SamplingSupportType sampling_strategy;
    static int group_size_;
    static int reuse_factor_;
    static bool postpone_nonlocal_;

    /**
     * \brief Program options for sampling support
     */
    static void AddSamplingOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sampling.strategy", po::value<SamplingSupportType>(&sampling_strategy)->default_value(SamplingSupportType::Naive), "Sampling strategy (naive [default], preloc, pool, or onlylocal)")
        ("sampling.group_size", po::value<int>(&group_size_)->default_value(250), "(only for 'pool' strategy) Group size, i.e., how may keys should be in one localize")
        ("sampling.reuse", po::value<int>(&reuse_factor_)->default_value(1), "(only for 'pool' strategy) Reuse factor: how often should a sample be used")
        ("sampling.postpone", po::value<bool>(&postpone_nonlocal_)->default_value(true), "(only for 'pool' strategy) Postpone non-local samples? (default: true)")
        ;
    }
  };



  /**
   * \brief Naive sampling support: choose any key and pull it
   */
  template <typename Val, typename Worker>
  class NaiveSamplingSupport : public SamplingSupport<Val, Worker> {
  public:
    NaiveSamplingSupport(Key (*const sample_key)()): SamplingSupport<Val, Worker>(sample_key) {
      ALOG("Naive sampling support");
    }
    using SamplingSupport<Val, Worker>::sample_key_;
    using SamplingSupport<Val, Worker>::id_counters;
    using SamplingSupport<Val, Worker>::workers;

    // naive prepare: do nothing
    SampleID prepare_sample(size_t K, int customer_id) override {
      return id_counters[customer_id-1]++;
    }

    // naive pull: sample randomly + pull
    int pull_sample(SampleID sample_id, std::vector<Key>& keys, std::vector<Val>& vals, const int customer_id) override {
      for(size_t k=0; k!=keys.size(); ++k) {
        keys[k] = (*sample_key_)();
      }

      SamplingSupport<Val, Worker>::count_accesses(keys);
      return workers[customer_id-1]->Pull(keys, &vals);
    };
  };



  /**
   * \brief Preloc sampling support: pre-localize keys in prep(), then pull those keys in pull()
   */
  template <typename Val, typename Worker>
  class PrelocSamplingSupport : public SamplingSupport<Val, Worker> {
  public:
    PrelocSamplingSupport(Key (*const sample_key)()): SamplingSupport<Val, Worker>(sample_key),
                                                      samples(Postoffice::Get()->num_worker_threads()),
                                                      num_pulls(Postoffice::Get()->num_worker_threads(), 0),
                                                      num_pulls_local(Postoffice::Get()->num_worker_threads(), 0),
                                                      num_preps(Postoffice::Get()->num_worker_threads(), 0),
                                                      num_keys(Postoffice::Get()->num_worker_threads(), 0),
                                                      num_unused(Postoffice::Get()->num_worker_threads(), 0),
                                                      num_unnecessary_localizes(Postoffice::Get()->num_worker_threads(), 0) {
      ALOG("Preloc sampling support");
    }
    using SamplingSupport<Val, Worker>::sample_key_;
    using SamplingSupport<Val, Worker>::id_counters;
    using SamplingSupport<Val, Worker>::workers;

    // preloc prep: sample keys and pre-localize them
    SampleID prepare_sample(size_t K, int customer_id) override {
      SampleID sample_id = id_counters[customer_id-1]++;
      PrelocSample sample;
      sample.keys.resize(K);
      for(size_t k=0; k!=K; ++k) {
        sample.keys[k] = (*sample_key_)();
      }

      // pre-localize the sampled keys (so that they are local when they will be accessed)
      size_t numLocal = 0;
      workers[customer_id-1]->Localize(sample.keys, &numLocal);

      // stats
      num_preps[customer_id-1]++;
      num_keys[customer_id-1] += K;
      if (numLocal == sample.keys.size()) num_unnecessary_localizes[customer_id-1]++;

      // store the sample until the pull call
      samples[customer_id-1][sample_id] = std::move(sample);

      return sample_id;
    }

    // preloc pull: pull the pre-localized keys
    int pull_sample(SampleID sample_id, std::vector<Key>& keys, std::vector<Val>& vals, const int customer_id) override {
      // find stored information for this sample
      auto search = samples[customer_id-1].find(sample_id);
      if (search == samples[customer_id-1].end()) {
        ALOG("Invalid sample sample_id: " << Postoffice::Get()->my_rank() << "::" << (customer_id-1) << "::" << sample_id);
        abort();
      }
      PrelocSample& sample = search->second;

      // use as many keys of this sample as the passed `keys` has space
      for(size_t k=0; k!=keys.size(); ++k, ++sample.used) {
        keys[k] = sample.keys[sample.used];
      }

      // clean up the memory if all keys of this sample have been used
      if (sample.used == sample.keys.size()) {
        samples[customer_id-1].erase(search);
      }

      SamplingSupport<Val, Worker>::count_accesses(keys);
      auto ts = workers[customer_id-1]->Pull(keys, &vals);
      num_pulls[customer_id-1]++; // stats
      if (ts == -1) num_pulls_local[customer_id-1]++; // stats
      return ts;
    };

    // clean up data about a sample, if it still exists
    void finish_sample(SampleID sample_id, int customer_id) override {
      auto search = samples[customer_id-1].find(sample_id);
      if (search != samples[customer_id-1].end()) {
        // delete information of the sample
        num_unused[customer_id-1] += search->second.keys.size() - search->second.used; // stats
        samples[customer_id-1].erase(search);
      }
    }

    // report the share of local pulls
    void terminate() override {
      unsigned long total = 0;
      unsigned long local = 0;
      for (auto n : num_pulls) total += n;
      for (auto n : num_pulls_local) local += n;
      unsigned long keys = 0;
      unsigned long unused = 0;
      for (auto n : num_keys) keys += n;
      for (auto n : num_unused) unused += n;
      unsigned long preps = 0;
      unsigned long unnecessary_localizes = 0;
      for (auto n : num_preps) preps += n;
      for (auto n : num_unnecessary_localizes) unnecessary_localizes += n;
      ALOG("Sampling " << Postoffice::Get()->my_rank() << ": " << total << " pulls, " << std::setprecision(5) << 100.0 * local / total << "% local, " << 100.0 * unnecessary_localizes / preps << "% of " << preps << " localizes unnecessary. " << 100.0 * unused / keys << "% of " << keys << " keys unused");
      SamplingSupport<Val, Worker>::terminate();
    }

  protected:

    // information storage (one per worker so we don't need to synchronize)
    std::vector<std::unordered_map<SampleID, PrelocSample>> samples;

    // stats
    std::vector<unsigned long> num_pulls;
    std::vector<unsigned long> num_pulls_local;
    std::vector<unsigned long> num_preps;
    std::vector<unsigned long> num_keys;
    std::vector<unsigned long> num_unused;
    std::vector<unsigned long> num_unnecessary_localizes;
  };


  /**
   * \brief Pool sampling support: each node holds one pool of keys, from which samples a drawn
   *        The pool is refreshed by a background thread.
   *        Additional parameters: the size of the pool and the reuse factor
   */
  template <typename Val, typename Worker, typename Server>
  class PoolSamplingSupport : public PrelocSamplingSupport<Val, Worker> {
    using SamplingSupport<Val, Worker>::sample_key_;
    using SamplingSupport<Val, Worker>::id_counters;
    using SamplingSupport<Val, Worker>::workers;
    using SamplingSupport<Val, Worker>::generator;
    using SamplingSupport<Val, Worker>::group_size_;
    using SamplingSupport<Val, Worker>::reuse_factor_;
    using SamplingSupport<Val, Worker>::postpone_nonlocal_;
    using PrelocSamplingSupport<Val, Worker>::num_preps;
    using PrelocSamplingSupport<Val, Worker>::num_keys;
    using PrelocSamplingSupport<Val, Worker>::num_unused;
    using PrelocSamplingSupport<Val, Worker>::num_pulls;
    using PrelocSamplingSupport<Val, Worker>::num_pulls_local;
    using PrelocSamplingSupport<Val, Worker>::num_unnecessary_localizes;


  public:
    PoolSamplingSupport(Key (*const sample_key)(), Server* server):
      PrelocSamplingSupport<Val, Worker>(sample_key),
      samples(Postoffice::Get()->num_worker_threads()),
      pool(group_size_*reuse_factor_*safety_factor*100), use_pos(0), prep_pos(0),
      server_(server),
      preps_total_elapsed_positions(group_size_*reuse_factor_), num_pool_preps(1) {
        ALOG("Pool sampling support. group size " << group_size_ << ", reuse factor " << reuse_factor_ << ", postpone " << postpone_nonlocal_);

        if (group_size_ == 0 || reuse_factor_ == 0) {
          ALOG("ERROR. Need to set pool size and reuse factor for pool sampling support.");
          abort();
        }
      }

    // pool prep: sample keys and pre-localize them
    SampleID prepare_sample(size_t K, int customer_id) override {
      SampleID sample_id = id_counters[customer_id-1]++;
      PoolSample sample;
      sample.keys.resize(K);
      sample.K = K;
      sample_keys_from_pool(sample.keys.begin(), sample.keys.end());

      // re-localize the sampled keys (in case another node took them away)
      size_t numLocal = 0;
      workers[customer_id-1]->Localize(sample.keys, &numLocal);

      // stats
      num_preps[customer_id-1]++;
      num_keys[customer_id-1] += K;
      if (numLocal == sample.keys.size()) num_unnecessary_localizes[customer_id-1]++;

      // store the sample until the pull call
      samples[customer_id-1][sample_id] = std::move(sample);

      return sample_id;
    }

    // pool pull
    int pull_sample(SampleID sample_id, std::vector<Key>& keys, std::vector<Val>& vals, const int customer_id) override {
      // find stored information for this sample
      auto search = samples[customer_id-1].find(sample_id);
      if (search == samples[customer_id-1].end()) {
        ALOG("Invalid sample sample_id: " << Postoffice::Get()->my_rank() << "::" << (customer_id-1) << "::" << sample_id);
        abort();
      }
      PoolSample& sample = search->second;

      int ts = 0;
      if (postpone_nonlocal_ && // postpone only if it is enabled
          keys.size() == 1 && // postponing is currently supported only for single-sample pulls
          sample.used < sample.K && // no postponing if we have reached previously postponed parameters already
          sample.used + 1 < sample.keys.size() // postpone only if there is a "next" key that we can use instead
          ) { // postponing: check whether key is local, postpone if it isn't

        // get key and check whether it is local
        keys[0] = sample.keys[sample.used];
        ++sample.used;
        auto local = workers[customer_id-1]->PullIfLocal(keys[0], &vals);

        if (local) { // if it is local, use it
          ts = -1;
        } else { // otherwise, we postpone this key and use the next one
          // postpone this key for later (to the end of this sample)
          sample.keys.push_back(keys[0]);
          ++num_postponed_total;

          // use the next key (and use this one even if it is not local)
          keys[0] = sample.keys[sample.used];
          ++sample.used;
          ts = workers[customer_id-1]->Pull(keys, &vals);
        }
      } else { // standard case: regular pull (no postponing)

        // use as many keys of this sample as the passed `keys` has space
        for(size_t k=0; k!=keys.size(); ++k, ++sample.used) {
          keys[k] = sample.keys[sample.used];
        }

        ts = workers[customer_id-1]->Pull(keys, &vals);
      }

      // clean up the memory if all keys of this sample have been used
      if (sample.used == sample.keys.size()) {
        samples[customer_id-1].erase(search);
      }

      num_pulls[customer_id-1]++; // stats
      if (ts == -1) num_pulls_local[customer_id-1]++; // stats
      SamplingSupport<Val, Worker>::count_accesses(keys);
      return ts;
    };

    // override so that we can spawn the mgr thread once workers have started
    void registerWorker (int customer_id, Worker* worker) override {
      SamplingSupport<Val,Worker>::registerWorker(customer_id, worker);

      // spawn pool manager thread
      if (customer_id == 1) {
        thread_  = std::thread (&PoolSamplingSupport<Val,Worker,Server>::thread, this);
        SET_THREAD_NAME((&thread_), "sampling");
      }
    }

    // clean up data about a sample, if it still exists
    void finish_sample(SampleID sample_id, int customer_id) override {
      auto search = samples[customer_id-1].find(sample_id);
      if (search != samples[customer_id-1].end()) {
        // delete information of the sample
        num_unused[customer_id-1] += search->second.keys.size() - search->second.used; // stats
        samples[customer_id-1].erase(search);
      }
    }

    // stop the pool manager thread
    void terminate () override {
      stop_thread = true;
      thread_.join();
      PrelocSamplingSupport<Val, Worker>::terminate();
    }

  private:
    // sample N keys from the current pool
    void sample_keys_from_pool(std::vector<Key>::iterator begin, std::vector<Key>::iterator end) {
      size_t num_keys = end-begin;
      auto pos_in_pool = use_pos.fetch_add(num_keys);

      // Sanity check: was the pool prep fast enough? If not: wait
      auto num_not_prepped_yet = (pos_in_pool+static_cast<long>(num_keys)) - prep_pos;
      if (num_not_prepped_yet > 0) {
        ALOG("[NOTE] Waiting for pool prep at rank " << Postoffice::Get()->my_rank() << ": use (" << use_pos << ") is " << num_not_prepped_yet << " positions ahead of prep (" << prep_pos << ")");

        // increase estimation of pool fetch time
        preps_total_elapsed_positions += num_not_prepped_yet * 2;

        while (pos_in_pool+static_cast<long>(num_keys) > prep_pos) {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      }

      // get samples
      for (size_t k=0; k!=num_keys; ++k) {
        *(begin+k) = pool[(pos_in_pool+k) % pool.size()];
      }
    }

    // prepare a new group of samples (i.e., sample new keys and localize them)
    // returns the timestamp of the localize call
    int prepare_new_group(const long pos, Worker& kv) {
      // sample a new group of parameters
      std::unordered_set<Key> to_localize {};
      for (int k=0; k!=group_size_; ++k) {
        pool[(pos+k) % pool.size()] = (*sample_key_)();
        to_localize.insert(pool[(pos+k) % pool.size()]);
      }

      // reuse > 1: copy the group in random order
      std::vector<size_t> target_positions (group_size_);
      for(int i = 1; i!=reuse_factor_; ++i) {
        // generate a random order
        std::iota(target_positions.begin(), target_positions.end(), pos+group_size_*i);
        std::random_shuffle(target_positions.begin(), target_positions.end());

        // copy with wrap-around
        for(int k=0; k!=group_size_; ++k) {
          pool[(target_positions[k]) % pool.size()] = pool[(pos+k) % pool.size()];
        }
      }

      // stats
      fetched_total += group_size_; // stats. don't recount the postponed

      return kv.Localize({to_localize.begin(), to_localize.end()});
    }

    // thread for refreshing the pool in the background
    // the thread is spawned once the worker with customer_id 1 has registered
    void thread() {
      auto my_rank = Postoffice::Get()->my_rank();
      ADLOG("Pool management thread " << my_rank << " started");

      // start our own worker
      int sampling_customer_id = Postoffice::Get()->num_worker_threads()+1;
      Postoffice::Get()->Start(sampling_customer_id, nullptr, true);
      Worker kv (0, sampling_customer_id, *server_);

      std::deque<std::pair<int, long>> ongoing_preps {};

      while (!stop_thread) {

        // prepare a new group of samples when necessary
        auto estimated_positions_for_fetch = 1.0 * (preps_total_elapsed_positions / num_pool_preps);
        if (use_pos + safety_factor * estimated_positions_for_fetch >=  prep_pos) {
          // sample new pool of keys and localize (asynchronously)
          auto pos = prep_pos.load(); // get position for new group
          int ts = prepare_new_group(pos, kv);
          ongoing_preps.push_back({ts, use_pos});
          prep_pos += group_size_ * reuse_factor_; // mark the new group as prepped
        } else {
          // wait to prevent busy loop
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // tune prep time estimate: check whether one of the prep localizes is finished
        if (!ongoing_preps.empty() && kv.IsFinished(ongoing_preps[0].first)) {
            auto pos_now = use_pos.load();
            if (use_pos != 0) { // don't count setup localizes
              ++num_pool_preps;
              preps_total_elapsed_positions += (pos_now - ongoing_preps[0].second);
            }
            ongoing_preps.pop_front();
        }

        // pool size check
        if (safety_factor * estimated_positions_for_fetch > pool.size()) {
          ADLOG("[ERROR] We currently estimate that it takes " << estimated_positions_for_fetch << " positions to fetch a group, but the pool has only " << pool.size() << " positions. With a safety factor of " << safety_factor << ", that is not enough.");
          abort();
        }
      }

      Postoffice::Get()->Finalize(sampling_customer_id, false);
      ALOG("Pool management thread " << my_rank << " stopped. Used " << use_pos << ", fetched " << fetched_total << ". (empiric reuse: " << std::setprecision(5) << 1.0 * use_pos / fetched_total << "). Postponed " << num_postponed_total << " (" << 100.0 * num_postponed_total/use_pos << "% of accesses)");
    }

    // store information about samples (for pool support)
    std::vector<std::unordered_map<SampleID, PoolSample>> samples;

    // Pool and positions
    std::vector<Key> pool; // we don't use a standard queue here because we don't want re-allocations
    std::atomic<long> use_pos;
    std::atomic<long> prep_pos;

    // settings
    static constexpr float safety_factor = 2.0;

    // pool manager thread
    std::thread thread_;
    bool stop_thread = false;

    Server* server_;

    // estimate of fetch time
    std::atomic<long> preps_total_elapsed_positions;
    long num_pool_preps;

    // stats
    unsigned long long fetched_total = 0;
    long num_postponed_total = 0;
  };


  /**
   * \brief "Only Local" sampling support: sample until we find a local parameter, then use that one.
   *        Note that this does not guarantee the sampling distribution.
   *        Also, the current implementation of the search for a local parameter
   *        is not particularly efficient.
   */
  template <typename Val, typename Worker>
  class OnlyLocalSamplingSupport : public NaiveSamplingSupport<Val, Worker> {
    using SamplingSupport<Val, Worker>::sample_key_;
    using SamplingSupport<Val, Worker>::id_counters;
    using SamplingSupport<Val, Worker>::workers;

  public:
    OnlyLocalSamplingSupport(Key (*const sample_key)()):
      NaiveSamplingSupport<Val, Worker>(sample_key), sampled_keys(Postoffice::Get()->num_worker_threads()) {
      ALOG("OnlyLocal sampling support");
    }

    // prepare_sample is identical to the naive one

    // OnlyLocal samples a lot of keys. To do this efficiently, we sample batches of Keys
    Key sample(const int customer_id) {
      constexpr size_t batch_size = 10000;
      // sample a new batch if necessary
      if (sampled_keys[customer_id-1].size() == 0) {
        for (size_t z=0; z!=batch_size; ++z) {
          sampled_keys[customer_id-1].push((*sample_key_)());
        }
      }

      auto key = sampled_keys[customer_id-1].front();
      sampled_keys[customer_id-1].pop();
      return key;
    }

    // only local pull: search for a local key, return it
    int pull_sample(SampleID sample_id, std::vector<Key>& keys, std::vector<Val>& vals, const int customer_id) override {
      assert(keys.size() == 1);
      ++num_pulls;
      bool found_neg = false;
      while (!found_neg) {
        keys[0] = sample(customer_id);
        found_neg = workers[customer_id-1]->PullIfLocal(keys[0], &vals);
        ++num_checks;
      }
      SamplingSupport<Val, Worker>::count_accesses(keys);
      return -1;
    }

    void terminate () override {
      ALOG("Only local sampling support at rank " << Postoffice::Get()->my_rank() << ": " << num_checks << " checks for " << num_pulls << " pulls (" << std::setprecision(3) << 1.0 * num_checks / num_pulls << " checks/pull)");
      SamplingSupport<Val, Worker>::terminate();
    }

  private:
    unsigned long long num_checks = 0;
    unsigned long long num_pulls = 0;
    std::vector<std::queue<Key>> sampled_keys;
  };


  // declare (static) program options
  template <typename Val, typename Worker> SamplingSupportType SamplingSupport<Val,Worker>::sampling_strategy;
  template <typename Val, typename Worker> int SamplingSupport<Val,Worker>::reuse_factor_;
  template <typename Val, typename Worker> int SamplingSupport<Val,Worker>::group_size_;
  template <typename Val, typename Worker> bool SamplingSupport<Val,Worker>::postpone_nonlocal_;
}

#endif  // PS_SAMPLING_H_

