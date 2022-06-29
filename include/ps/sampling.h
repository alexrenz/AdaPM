/**
 *  Sampling support in a parameter server
 *
 *  Implements different approaches for supporting sampling, i.e., pulling any
 *  of a specified group of keys, according to a specified distribution.
 *
 *  Currently implemented sampling schemes:
 *   - Naive
 *   - Pre-localization
 *   - Pool
 *   - Local
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

  // Available Schemes for Sampling Support
  enum class SamplingScheme {Naive, Preloc, Pool, Local};

  std::ostream& operator<<(std::ostream &o, const SamplingScheme& smt) {
    switch(smt) {
    case SamplingScheme::Naive: return o << "naive";
    case SamplingScheme::Preloc: return o << "preloc";
    case SamplingScheme::Pool: return o << "pool";
    case SamplingScheme::Local: return o << "local";
    default: return o << "unknown";
    }
  }
  std::istream& operator>>(std::istream& in, SamplingScheme& smt) {
    std::string token; in >> token;
    if (token == "naive") smt = SamplingScheme::Naive;
    else if (token == "preloc") smt = SamplingScheme::Preloc;
    else if (token == "pool") smt = SamplingScheme::Pool;
    else if (token == "local") smt = SamplingScheme::Local;
    else { CHECK(false) << "Fatal! Unknown sampling scheme '" << token << "'"; }
    return in;
  }


  // per-sample information that we store between the prep and the pull call
  struct SimpleSample {
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
  template <typename Val, typename Worker, typename Server>
    class Sampling {
  protected:
    std::default_random_engine generator;

    // pointer to (application-provided) function to sample 1 key
    Key (*const sample_key_)();

    // counters for creating unique sample IDs (one per customer)
    std::vector<size_t> id_counters;

    // pointer to the server
    Server* server_;

    // stats about number of non-deterministic accesses
#if PS_LOCALITY_STATS
    std::vector<unsigned long> num_accesses;
    int myRank;
#endif

    // count key accesses
    inline void count_accesses(const Key* keys, const size_t num_keys) {
#if PS_LOCALITY_STATS
      for (unsigned int i=0; i!=num_keys; ++i) {
        num_accesses[keys[i]]++;
      }
#endif
    }

  public:
    Sampling(Key (*const sample_key)(), Server* server):
      sample_key_(sample_key), id_counters(Postoffice::Get()->num_worker_threads(), 1),
      server_(server) {
#if PS_LOCALITY_STATS
      ALOG("Capture number of non-deterministic accesses per key");
      myRank = Postoffice::Get()->my_rank();
      num_accesses.resize(Postoffice::Get()->num_keys(), 0);
#endif
    }

    virtual ~Sampling() {};

    // draw `num_samples` random samples into `samples`, either with or without replacement
    virtual inline void draw_samples(Key* samples, const size_t num_samples) {
      if (with_replacement_) { // with replacement
        for(size_t k=0; k!=num_samples; ++k) {
          samples[k] = (*sample_key_)();
        }
      } else { // without replacement
        std::unordered_set<Key> samples_set (num_samples);
        while (samples_set.size() != num_samples) {
          samples_set.insert((*sample_key_)());
        }
        std::copy(samples_set.begin(), samples_set.end(), samples);
      }
    }

    // hook to terminate any activities
    virtual void terminate() {
#if PS_LOCALITY_STATS
      // write number of non-deterministic accesses to file
      std::string outfile ("stats/locality_stats.sampling.rank." + std::to_string(myRank) + ".tsv");
      std::ofstream statsfile (outfile, std::ofstream::trunc);
      long total_accesses = 0;
      statsfile << "Param\tAccesses\n";
      for (uint i=0; i!=num_accesses.size(); ++i) {
        statsfile << i << "\t" << num_accesses[i] << "\n";
        total_accesses += num_accesses[i];
      }
      statsfile.close();
      ALOG("Wrote non-deterministic access stats for rank " << myRank << " to " << outfile << ". Total: " << total_accesses << " accesses");
#endif
    }

    // prepare a sample of `K` keys; returns a sample ID
    virtual SampleID prepare_sample(size_t K, int customer_id, Clock start, Clock end) = 0;

    // pull `num_keys` keys from the prepared sample with ID `id`; returns an operation timestamp
    virtual int pull_sample(SampleID sample_id, Key* keys, const size_t num_keys, Val* vals, const int customer_id) = 0;

    // declare a sample as finished (so that any related data can be deleted)
    virtual void finish_sample(SampleID sample_id, int customer_id) {};

    // program options
    static SamplingScheme scheme;
    static int pool_size_;
    static unsigned int reuse_factor_;
    static size_t sampling_batch_size_;
    static bool with_replacement_;

    /**
     * \brief Program options for sampling support
     */
    static void AddSamplingOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sampling.scheme", po::value<SamplingScheme>(&scheme)->default_value(SamplingScheme::Local), "Sampling scheme (naive, preloc, pool, or local [default])")
        ("sampling.pool_size", po::value<int>(&pool_size_)->default_value(250), "(only for 'pool' scheme) Pool size. I.e., how many samples should be in the pool")
        ("sampling.reuse", po::value<unsigned int>(&reuse_factor_)->default_value(1), "(only for 'pool' scheme) Reuse factor: how often should a sample be used")
        ("sampling.batch_size", po::value<size_t>(&sampling_batch_size_)->default_value(10000), "In local sampling we batch draw samples in batches. This determines the batch size (default: 10000). Usually, you should not need to change this value.")
        ("sampling.with_replacement", po::value<bool>(&with_replacement_)->default_value(true), "Whether to sample with replacement (default) or without 'replacement")
        ;
    }
  };



  /**
   * \brief Naive sampling support: choose any key and pull it
   */
  template <typename Val, typename Worker, typename Server>
  class NaiveSampling : public Sampling<Val, Worker, Server> {
  public:
    NaiveSampling(Key (*const sample_key)(), Server* server):
      Sampling<Val, Worker, Server>(sample_key, server),
      samples(Postoffice::Get()->num_worker_threads()) {
      ALOG("Naive sampling support");
    }
    using Sampling<Val, Worker, Server>::sample_key_;
    using Sampling<Val, Worker, Server>::id_counters;
    using Sampling<Val, Worker, Server>::server_;
    using Sampling<Val, Worker, Server>::draw_samples;

    // naive prepare: do nothing
    SampleID prepare_sample(size_t K, int customer_id, Clock start, Clock end) override {
      auto sample_id = id_counters[customer_id]++;

      SimpleSample sample;
      sample.keys.resize(K);
      draw_samples(sample.keys.data(), K);
      samples[customer_id][sample_id] = std::move(sample);

      return sample_id;
    }

    // naive pull: sample randomly + pull
    int pull_sample(SampleID sample_id, Key* keys, const size_t num_keys,
                    Val* vals, const int customer_id) override {

      auto search = samples[customer_id].find(sample_id);
      if (search == samples[customer_id].end()) {
        ALOG("Invalid sample sample_id: " << Postoffice::Get()->my_rank() << "::" << (customer_id) << "::" << sample_id);
        abort();
      }
      SimpleSample& sample = search->second;

      // use as many keys of this sample as the passed `keys` has space
      for(size_t k=0; k!=num_keys; ++k, ++sample.used) {
        keys[k] = sample.keys[sample.used];
      }

      // clean up the memory if all keys of this sample have been used
      if (sample.used == sample.keys.size()) {
        samples[customer_id].erase(search);
      }

      Sampling<Val, Worker, Server>::count_accesses(keys, num_keys);
      return server_->getWorker(customer_id)->Pull(keys, num_keys, vals);
    };

    // clean up data about a sample, if it still exists
    void finish_sample(SampleID sample_id, int customer_id) override {
      auto search = samples[customer_id].find(sample_id);
      if (search != samples[customer_id].end()) {
        // delete information of the sample
        samples[customer_id].erase(search);
      }
    }

    // sample storage (one per worker so we don't need to synchronize)
    std::vector<std::unordered_map<SampleID, SimpleSample>> samples;
  };



  /**
   * \brief Preloc sampling support: pre-localize keys in prep(), then pull those keys in pull()
   */
  template <typename Val, typename Worker, typename Server>
  class PrelocSampling : public NaiveSampling<Val, Worker, Server> {
  public:
    PrelocSampling(Key (*const sample_key)(), Server* server):
      NaiveSampling<Val, Worker, Server>(sample_key, server) {
      ALOG("Preloc sampling support");
    }
    using Sampling<Val, Worker, Server>::sample_key_;
    using Sampling<Val, Worker, Server>::id_counters;
    using Sampling<Val, Worker, Server>::server_;
    using Sampling<Val, Worker, Server>::draw_samples;
    using NaiveSampling<Val, Worker, Server>::samples;

    // preloc prep: sample keys and pre-localize them
    SampleID prepare_sample(size_t K, int customer_id, Clock start, Clock end) override {
      SampleID sample_id = id_counters[customer_id]++;
      SimpleSample sample;
      sample.keys.resize(K);
      draw_samples(sample.keys.data(), K);

      // signal intent for the sampled keys
      server_->getWorker(customer_id)->Intent(sample.keys, start, end);

      // store the sample until the pull call
      samples[customer_id][sample_id] = std::move(sample);

      return sample_id;
    }

    // preloc pull: identical to naive

    // preloc finish: identical to naive
  };


  /**
   * \brief Pool sampling support: each node holds one pool of keys, from which samples a drawn
   *        The pool is refreshed by a background thread.
   *        Additional parameters: the size of the pool and the reuse factor
   */
  template <typename Val, typename Worker, typename Server>
  class PoolSampling : public PrelocSampling<Val, Worker, Server> {
    using Sampling<Val, Worker, Server>::sample_key_;
    using Sampling<Val, Worker, Server>::id_counters;
    using Sampling<Val, Worker, Server>::server_;
    using Sampling<Val, Worker, Server>::generator;
    using Sampling<Val, Worker, Server>::pool_size_;
    using Sampling<Val, Worker, Server>::reuse_factor_;
    using Sampling<Val, Worker, Server>::with_replacement_;
    using PrelocSampling<Val, Worker, Server>::samples;


  public:
    PoolSampling(Key (*const sample_key)(), Server* server):
      PrelocSampling<Val, Worker, Server>(sample_key, server),
      pool(pool_size_), pool_pos(pool.size()), pool_num_uses(reuse_factor_) {
        ALOG("Pool sampling support. pool size " << pool_size_ << ", reuse factor " << reuse_factor_);

        if (pool_size_ == 0 || reuse_factor_ == 0) {
          ALOG("ERROR. Need to set pool size and reuse factor for pool sampling support.");
          abort();
        }

        if (!with_replacement_) {
          ALOG("Not implemented yet: Pool sampling with 'without replacement' sampling");
          abort();
        }
      }



  private:
    // draw pool samples into `v`
    inline void draw_samples(std::vector<Key>& v) {
      std::lock_guard<std::mutex> lk (pool_mu_);

      for (size_t k=0; k!=v.size(); ++k) {

        // do we need to refresh the pool?
        if (pool_pos >= pool.size()) {
          ++pool_num_uses;

          if (pool_num_uses < reuse_factor_) {
            std::random_shuffle(pool.begin(), pool.end());
          } else {
            sample_new_pool();
            pool_num_uses = 0;
          }

          pool_pos = 0;
        }

        v[k] = pool[pool_pos];
        ++pool_pos;
      }
    }

    // fill the pool with new samples (unsafe)
    void sample_new_pool() {
      for (size_t k=0; k!=pool.size(); ++k) {
        pool[k] = (*sample_key_)();
      }
    }

    // Pool and positions
    std::vector<Key> pool;
    size_t pool_pos;
    size_t pool_num_uses;
    std::mutex pool_mu_;
  };


  /**
   * \brief Local sampling: sample until we find a local parameter, then use that one.
   *        Note that this does not guarantee the sampling distribution.
   *        Also, the current implementation of the search for a local parameter
   *        is not particularly efficient.
   */
  template <typename Val, typename Worker, typename Server>
  class LocalSampling : public NaiveSampling<Val, Worker, Server> {
    using Sampling<Val, Worker, Server>::sample_key_;
    using Sampling<Val, Worker, Server>::id_counters;
    using Sampling<Val, Worker, Server>::server_;
    using Sampling<Val, Worker, Server>::sampling_batch_size_;
    using Sampling<Val, Worker, Server>::with_replacement_;

  public:
  LocalSampling(Key (*const sample_key)(), Server* server, const Key min_key, const Key max_key):
      NaiveSampling<Val, Worker, Server>(sample_key, server),
      sampled_keys(Postoffice::Get()->num_worker_threads()),
      keys_in_sample(Postoffice::Get()->num_worker_threads()),
      _min_key(min_key), _max_key(max_key) {
      ALOG("Local sampling, " << (_min_key == _max_key ? "general" : "memory friendly") << " implementation");
    }

    SampleID prepare_sample(size_t K, int customer_id, Clock start, Clock end) override {
      SampleID sample_id = id_counters[customer_id]++;

      if (!with_replacement_) {
        keys_in_sample[customer_id].emplace(sample_id, std::unordered_set<Key>(K));
      }

      return sample_id;
    }

    // Local sampling samples a lot of keys. To do this efficiently, we sample batches of Keys
    Key sample(const int customer_id) {
      // sample a new batch if necessary
      if (sampled_keys[customer_id].size() == 0) {
        for (size_t z=0; z!=sampling_batch_size_; ++z) {
          sampled_keys[customer_id].push((*sample_key_)());
        }
      }

      auto key = sampled_keys[customer_id].front();
      sampled_keys[customer_id].pop();
      return key;
    }

    // local sampling pull: search for a local key, return it
    int pull_sample(SampleID sample_id, Key* keys, const size_t num_keys,
                    Val* vals, const int customer_id) override {
      ++num_pulls;

      size_t pos = 0;

      // for WOR sampling: get a pointer to the set of keys that were already sampled in this sample
      std::unordered_set<Key>* already_sampled;
      if (!with_replacement_) {
        auto search = keys_in_sample[customer_id].find(sample_id);
        if (search == keys_in_sample[customer_id].end()) {
          ALOG("Invalid sample sample_id: " << Postoffice::Get()->my_rank() << "::" << (customer_id) << "::" << sample_id);
          abort();
        }
        already_sampled = &(search->second);
      }

      // GENERAL IMPLEMENTATION (sampling range is not guaranteed to be continuous)
      if (_min_key == _max_key) {
        for (size_t i=0; i!=num_keys; ++i) {
          bool found_neg = false;
          while (!found_neg) {
            keys[i] = sample(customer_id);

            // if we do WOR sampling: check that the key has not been sampled before
            if (!with_replacement_ &&
                already_sampled->find(keys[i]) != already_sampled->end()) {
              continue; // this key was sampled already -> try a new one
            }

            found_neg = server_->getWorker(customer_id)->PullIfLocal(keys[i], vals+pos);
            ++num_checks;
          }
          pos += server_->GetLen(keys[i]);

          if (!with_replacement_) {
            already_sampled->insert(keys[i]);
          }
        }

      // MEMORY-FRIENDLY IMPLEMENTATION (used if the sampling range is guaranteed to be continuous)
      } else {
        for (size_t i=0; i!=num_keys; ++i) {
          Key start_key = sample(customer_id);
          keys[i] = pull_next_local(start_key, vals+pos, _min_key, _max_key, num_checks,
                                    with_replacement_, already_sampled);
          if (!with_replacement_) {
            already_sampled->insert(keys[i]);
          }
          pos += server_->GetLen(keys[i]);
        }
      }

      Sampling<Val, Worker, Server>::count_accesses(keys, num_keys);
      return -1;
    }

    // Starting from `key`, pull the next local key, searching upwards. I.e.,
    // if `key` is local, that key's values are pulled into `vals`; o/w, if key
    // `key`+1 is local, that key is pulled; o/w, if key `key`+2 is local, that
    // key is pulled; and so on ...
    //
    // The search wraps around to `min_key` when `max_key` is reached.
    //
    // If without-replacement sampling is used (with_replacement=false), keys in
    // `already_sampled` are skipped.
    //
    // Returns the key that was pulled
    Key pull_next_local(Key key, Val* vals,
                        const Key min_key, const Key max_key,
                        unsigned long long& num_checks,
                        const bool with_replacement,
                        std::unordered_set<Key>* already_sampled) {
      while (true) {
        // if we do without-replacement sampling: check whether key has been
        // sampled before (if it has, we move to the next key right away)
        if (with_replacement ||
            already_sampled->find(key) == already_sampled->end()) {

          // check whether this key is local
          if (!server_->request_handle_.isNonLocal_noLock(key)) {
            // key is probably local, try to pull it
            bool local = server_->request_handle_.attemptLocalPull(key, vals, false);
            ++num_checks;

            if (local) {
              return key;
            }
          }
        }

        // this key was not local, move on to the next one
        ++key;
        if (key >= max_key) {
          key = min_key;
        }
      }
    }

    void finish_sample(SampleID sample_id, int customer_id) override {
      if (!with_replacement_) {
        keys_in_sample[customer_id].erase(sample_id);
      }
    }

    void terminate () override {
      ALOG("Local sampling at rank " << Postoffice::Get()->my_rank() << ": " << num_checks << " checks for " << num_pulls << " pulls (" << std::setprecision(3) << 1.0 * num_checks / num_pulls << " checks/pull)");
      Sampling<Val, Worker, Server>::terminate();
    }

  private:
    unsigned long long num_checks = 0;
    unsigned long long num_pulls = 0;
    std::vector<std::queue<Key>> sampled_keys;
    std::vector<std::unordered_map<SampleID, std::unordered_set<Key>>> keys_in_sample;
    const Key _min_key;
    const Key _max_key;
  };


  // declare (static) program options
  template <typename Val, typename Worker, typename Server> SamplingScheme Sampling<Val, Worker, Server>::scheme = SamplingScheme::Local;
  template <typename Val, typename Worker, typename Server> unsigned int Sampling<Val, Worker, Server>::reuse_factor_ = 1;
  template <typename Val, typename Worker, typename Server> int Sampling<Val, Worker, Server>::pool_size_ = 250;
  template <typename Val, typename Worker, typename Server> size_t Sampling<Val, Worker, Server>::sampling_batch_size_ = 10000;
  template <typename Val, typename Worker, typename Server> bool Sampling<Val, Worker, Server>::with_replacement_ = true;
}

#endif  // PS_SAMPLING_H_

