/**
 *  Copyright (c) 2020 Alexander Renz-Wieland
 */
#ifndef PS_SYNC_MANAGER_H_
#define PS_SYNC_MANAGER_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <valarray>
#include <unordered_map>
#include <ps/internal/postoffice.h>
#include <zmq_van.h>
#include "utils.h"
#include "ps/kv_app.h"
#include <math.h>
#include <boost/program_options.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <sstream>

namespace ps {
  const Key SHUTDOWN = -1;

  template<typename Val>
  struct SyncData {
    KVPairs<Val> kvs;
    boost::dynamic_bitset<> includes_update;
    boost::dynamic_bitset<> refresh_desired;
    boost::dynamic_bitset<>& relocate {refresh_desired}; // alias, used in responses
    SArray<Version> versions;
    SArray<Key> inform_relocations; // empty in first msg, used in forwards to tell home node about relocations
    SArray<Version> relocation_counters;

    // pre-allocate memory to prevent repeated copying when constructing msgs (especially for vals)
    void reserve(size_t num_keys, size_t num_vals) {
      kvs.keys.reserve(num_keys);
      versions.reserve(num_keys);
#if BOOST_VERSION >= 106200 // dynamic_bitset::reserve() is provided only in recent boost versions
      includes_update.reserve(num_keys);
      refresh_desired.reserve(num_keys);
#endif

      kvs.vals.reserve(num_vals);
    }
  };

  /**
   * \brief Maintains and tunes an estimate for when the system should act on intents.
   *
   *        This tuner estimates how many clocks the workers will pass until the
   *        next sync round. It then tunes the estimate based on whether the
   *        workers passed more or fewer clocks than what it estimated.
   */
  class ActionTimer {
  public:
    ActionTimer ():
      noc_estimates(Postoffice::Get()->num_worker_threads(), initial_noc_estimate),
      last_round_worker_clocks(Postoffice::Get()->num_worker_threads()) {}

    std::vector<Clock> estimate_sync_windows_and_tune (const std::vector<Clock>& worker_clocks,
                                                       const unsigned long sync_round,
                                                       const unsigned int channel) {
      // without tuning: just use the initial estimate
      std::vector<Clock> sync_windows (Postoffice::Get()->num_worker_threads());

      // obtain (and tune) estimates
      for (size_t w=0; w!=Postoffice::Get()->num_worker_threads(); ++w) {

        // without tuning
        float estimate_for_this_round = noc_estimates[w];

        // with tuning
        if (sync_round > 0) {
          auto worker_ticks = static_cast<float>(worker_clocks[w] - last_round_worker_clocks[w]);
          if (tuning_enabled &&
              worker_clocks[w] != WORKER_FINISHED && worker_ticks != 0) {

            // update the estimate (using exponential smoothing)
            noc_estimates[w] = (1-smoothing_factor)*(noc_estimates[w]) + (smoothing_factor)*(worker_ticks);

            // never estimate fewer clocks than we saw the worker do in the last round
            estimate_for_this_round = std::max(worker_ticks, noc_estimates[w]);
          }
        }

        // estimate the sync window size based on the current estimate: 2 rounds + buffer
        if (estimate_for_this_round < 1000000) {
          auto poiss = boost::math::poisson_distribution<float>(2*estimate_for_this_round);
          sync_windows[w] = boost::math::quantile(poiss, buffer_quantile);
        } else {
          // boost's poisson implementation does not work for large lambdas, so
          // we approximate poisson with a normal distribution (very close for large lambdas):
          // Poisson(lambda) => N(lambda,sqrt(lambda))
          auto norm = boost::math::normal_distribution<>(2*estimate_for_this_round, std::sqrt(2*estimate_for_this_round));
          sync_windows[w] = boost::math::quantile(norm, buffer_quantile);
        }
      }

      // note down the current worker clocks (used for tuning in the next round)
      std::copy(worker_clocks.begin(), worker_clocks.end(), last_round_worker_clocks.begin());

      return sync_windows;
    }

    // obtain the current estimate (averaged over all workers)
    float get_avg_estimate() const {
      float sum_estimates = 0;
      for(auto est : noc_estimates) {
        sum_estimates += est;
      }
      return sum_estimates / noc_estimates.size();
    }

    // tuning program options
    static void AddProgramOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sys.timing.initial_estimate", po::value<float>(&initial_noc_estimate)->default_value(10), "Initial value for the 'number of clocks in sync round' estimate")
        ("sys.timing.autotune", po::value<bool>(&tuning_enabled)->default_value(true), "Automatically tune the 'number of clocks' estimate? (default: yes)")
        ("sys.timing.smoothing_factor", po::value<float>(&smoothing_factor)->default_value(0.1), "The smoothing factor `alpha` with which to mix the number of clocks in the last sync round into the estimate [new_estimate = (1-alpha)*(old_estimate) + (alpha)*(num_clocks_last_round)]")
        ("sys.timing.buffer_quantile", po::value<float>(&buffer_quantile)->default_value(0.9999), "The quantile for the poisson distribution that determines the safety buffer that we add to the estimate to determine the sync window.")
        ;
    }

    // output
    std::string DebugStr() {
      std::stringstream ss;
      ss << get_avg_estimate() << " avg";
      if (tuning_enabled) {
        ss << ", tune with smoothing factor " << smoothing_factor;
      } else {
        ss << ", no tuning";
      }
      return ss.str();
    }

  private:

    // "number of clocks" estimate: how many clocks does one worker do during one sync round?
    std::vector<float> noc_estimates;

    // tuning options (see description in `AddProgramOptions`)
    static bool tuning_enabled;
    static float initial_noc_estimate;
    static float smoothing_factor;
    static float buffer_quantile;

    // holds the worker clocks at the start of the last sync round
    std::vector<Clock> last_round_worker_clocks;
  };

  // default values for static members (repeated here so that they are set in case `AddProgramOptions` is not called)
  float ActionTimer::initial_noc_estimate = 10;
  bool ActionTimer::tuning_enabled = true;
  float ActionTimer::smoothing_factor = 0.1;
  float ActionTimer::buffer_quantile = 0.9999;


  template <typename Val, typename Handle>
  class SyncManager {

    using ServerT = ColoKVServer<Val, Handle>;

    // threshold for selective synchronization (see program options for description)
    static double sync_threshold;

    // channel
    unsigned int my_channel;

    // rank of this sync manager
    unsigned int my_rank;

    // total number of sync managers
    int world_size;

    // reference to the parameter store
    Handle* handle;

    // (approximate) intent of nodes
    std::vector<std::unordered_set<Key>> node_intent;

    // counter for synchronization rounds
    std::atomic<unsigned long> syncs {0};
    unsigned long syncs_since_last_report = 0;
    std::atomic<unsigned long> num_responses {0};
    std::atomic<unsigned long> num_refreshes_requested {0};
    std::atomic<unsigned long> num_refreshes_answered  {0};
    unsigned long total_refresh_hops {0}; // no atomic, only the PS thread edits this
    std::vector<unsigned long> hop_stats;

    // pointer to the server
    ServerT* server;

    // a timer that estimates how far into the future to act on intents
    ActionTimer action_timer;

    // stores new intents, grouped by key. this is a class variable because we reuse it across calls
    std::unordered_map<Key,std::vector<WorkerClockPair>> new_relevant_intents {};

    // sync manager thread terminates as soon as this is set to true
    bool terminate = false;

    // waiting between synchronization rounds
    std::function<void()> wait_function_; // wait function (multiple options available)
    static int sync_pause; // period pause (in milliseconds)
    static double syncs_per_sec; // aim for specific number of syncs per seconds
    std::chrono::milliseconds sync_interval {}; // interval (calculated, for interval pauses)
    std::chrono::time_point<std::chrono::high_resolution_clock> last_run; // last run (for interval pauses)

    // stopwatches
    util::Stopwatch sw_runtime, time_since_last_report, sw_pausing, sw_finishing, sw_collecting;

    // stats
    long updated = 0;
    long total = 0;
    long updated_since_last_report = 0;
    long total_since_last_report = 0;
    long long numRefreshsRequested = 0;
    long long numRefreshsNonZero = 0;
    long num_replicate = 0;
    long num_relocate = 0;
    Clock ticks_at_last_report = 0;

    /**
       Send out local replica updates (to the keys' manager)
     */
    void sendSyncRequest (const int destination, const SyncData<Val>& req,
                          const int message_trace, const int hops,
                          const bool forward=false, const int setSender=-1) {
      Message msg;
      msg.meta.app_id = message_trace; // NOTE: for debugging, abuse the app_id field for a trace
      msg.meta.hops = hops;
      msg.meta.channel = my_channel;
      msg.meta.customer_id = 0;
      msg.meta.request     = true;
      msg.meta.head        = (forward ? Control::SYNC_FORWARD : Control::SYNC);
      msg.meta.recver      = Postoffice::Get()->ServerRankToID(destination);
      if (setSender != -1) {
        msg.meta.sender      = setSender;
      }

      assert(req.refresh_desired.count() == req.versions.size());
      msg.AddData(req.kvs.keys);
      msg.AddData(req.kvs.vals);
      msg.AddData(req.includes_update);
      msg.AddData(req.refresh_desired);
      msg.AddData(req.versions);
      msg.AddData(req.inform_relocations); // empty in first message
      msg.AddData(req.relocation_counters); // empty in first message

      Postoffice::Get()->van()->Send(msg);
    }

    // look for new relevant intents and register them in the handle
    void registerNewIntents(const std::vector<Clock>& worker_clocks, const std::vector<Clock>& sync_window_sizes) {
      // get intents from the workers. we group intents by key so we need to
      // touch each key only once in the next step
      new_relevant_intents.clear(); // we reuse the map so that it has an appropriate size after some calls
      for (size_t w=0; w!=Postoffice::Get()->num_worker_threads(); ++w) {
        if (server->getWorker(w) != nullptr) { // skip workers that have shut down already
          server->getWorker(w)->getNewRelevantIntents(my_channel,
                                                      worker_clocks[w],
                                                      worker_clocks[w]+sync_window_sizes[w],
                                                      new_relevant_intents);
        }
      }

      // register intents in the handle
      server->request_handle_.lockAll();
      for (auto& elem : new_relevant_intents) {
        Key key = elem.first;

        // check that key is in key range
        CHECK(key < Postoffice::Get()->num_keys()) <<
          "[ERROR] Intent for key " << key << ", which is outside the configured " <<
          "key range [0,"<< Postoffice::Get()->num_keys() << ")";

        // note down intent for this key
        server->request_handle_.lockSingle(key);
        server->request_handle_.registerNewIntentsForKeyUnsafe(key, elem.second);
        server->request_handle_.unlockSingle(key);
      }
      server->request_handle_.unlockAll();
    }

    /**
     * \brief Synchronize updates among nodes
     */
    int startSync () {
      sw_pausing.stop();
      sw_finishing.resume();

      // wait for last sync round to finish
      while (num_refreshes_answered < num_refreshes_requested) {
        if (terminate) { // stop waiting if we want to terminate
          return 0;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      sw_finishing.stop();
      sw_collecting.resume();

      std::vector<SyncData<Val>> messages (Postoffice::Get()->num_servers());

      auto worker_clocks = server->getWorkerClocks();

      // If action timing is disabled, we act on all intents that have been communicated so far.
      // To achieve that, we set the maximum window
      std::vector<Clock> sync_window_sizes (Postoffice::Get()->num_worker_threads(), WINDOW_MAX);

      // If action timing is enabled, the action timer estimates what the window size should be.
      // We pass in current worker clocks so the action timer can tune its estimate.
      if (Postoffice::Get()->time_intent_actions()) {
        sync_window_sizes = action_timer.estimate_sync_windows_and_tune(worker_clocks, syncs.load(), my_channel);
      }

      // look for new relevant intents and register them in the handle
      registerNewIntents(worker_clocks, sync_window_sizes);

      // get list of current replicas
      auto current_replicas = handle->currentReplicas(my_channel);
      std::sort(current_replicas.begin(), current_replicas.end());

      // pre-allocate memory for the update messages
      auto total_len = handle->get_total_len(current_replicas);
      for(size_t z=0; z!=messages.size(); ++z) {
        messages[z].reserve(current_replicas.size() / messages.size(),
                            total_len / messages.size());
      }

      // go through the replicas to extract updates (also drop replicas if not needed anymore)
      for (unsigned long i=0; i!=current_replicas.size(); ++i) {
        Key key = current_replicas[i];
        auto destination = server->addressbook.getDirections(key, true); // use location cache
        bool refresh;
        bool extracted_an_update;
        Version ver;
        std::tie(refresh, extracted_an_update, ver) =
          handle->readAndPotentiallyDropReplica(key, messages[destination].kvs, worker_clocks, sync_threshold);

        // add this key and its information to the message
        if (refresh || extracted_an_update) {
          messages[destination].kvs.keys.push_back(key);
          messages[destination].refresh_desired.push_back(refresh);
          messages[destination].includes_update.push_back(extracted_an_update);

          if (refresh) {
            messages[destination].versions.push_back(ver);
            ++num_refreshes_requested;
          }
        }
      }

      // stats
      total += current_replicas.size();
      total_since_last_report += current_replicas.size();

      // send out updates
      for (unsigned int rank = 0; rank!=messages.size(); ++rank) {
        // Create a message trace with useful information (for debugging)
        int message_trace = syncs * 1e4 + my_rank * 1e2 + rank;

        // send messages only to other servers
        if (rank != my_rank) {
          sendSyncRequest(rank, messages[rank], message_trace, 1);
        } else {
          assert(messages[rank].kvs.keys.size() == 0); // there message to the node itself should be empty
        }
        // stats
        updated += messages[rank].kvs.keys.size();
        updated_since_last_report += messages[rank].kvs.keys.size();
      }

      ++syncs;
      ++syncs_since_last_report;

      sw_collecting.stop();
      sw_pausing.resume();
      return 0;
    }

    // no wait between two synchronization rounds
    void wait_none() {}

    // wait for a specified amount of time between synchronization rounds
    void wait_period() {
      if (syncs == 0) {
        return;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(sync_pause));
      }
    }

    // synchronize on intervals (e.g., 5x per second)
    void wait_interval() {
      if (syncs != 0) { // (don't wait on first call)
        auto now = std::chrono::high_resolution_clock::now();
        auto target = last_run + sync_interval;

        if (now > target) {
          // if interval already passed: don't do anything
        } else {
          // else: wait until interval comes again
          std::this_thread::sleep_until (target);
        }
      }

      last_run = std::chrono::high_resolution_clock::now();
    }

  public:
    /**
     * \brief Construct sync manager object
     */
    SyncManager() :
      node_intent(Postoffice::Get()->num_servers()), hop_stats(30), action_timer() {
      my_rank = Postoffice::Get()->my_rank();
      world_size = Postoffice::Get()->num_servers();

      // set wait method
      assert(sync_pause == 0 || syncs_per_sec == 0);
      if (sync_pause == 0 && syncs_per_sec == 0)
        wait_function_ = std::bind(&SyncManager::wait_none, this);
      else if (sync_pause != 0)
        wait_function_ = std::bind(&SyncManager::wait_period, this);
      else {
        assert(syncs_per_sec >= 0);
        sync_interval = std::chrono::milliseconds(static_cast<size_t>(1000 / syncs_per_sec));
        wait_function_ = std::bind(&SyncManager::wait_interval, this);
      }

      sw_pausing.start();
    }

    /** finish setup */
    void init (unsigned int channel, Handle* h, ServerT* s) {
      my_channel = channel;
      handle = h;
      server = s;
    }

    /** Get current sync round of this sync manager */
    unsigned long GetCurrentSync() {
      return syncs;
    }

    /**
     * \brief Separate Sync Manager thread that synchronizes nodes in the background
     */
    void thread () {
      // do nothing if there is only one process
      if (world_size == 1) {
        ALOG("Sync Manager: don't start as there is only one process globally.");
        return;
      }

      ALOG("SM" << my_rank << Postoffice::Get()->fchannel(my_channel) << " set up. Starting sync. " <<
           "Tech: " << server->management_techniques_str() <<
           ". Location caches: " << Postoffice::Get()->use_location_caches() <<
           ". Noc estimate avg: " << action_timer.DebugStr() <<
           ". Time intent: " << Postoffice::Get()->time_intent_actions());
      sw_runtime.start();
      time_since_last_report.start();

      // initial wait (for setup mostly)
      wait_function_();

      // sync manager background loop
      if (!std::isinf(sync_threshold)) { // threshold=inf disables sync entirely
        while (true) {
          if (terminate) {
            ALOG("Terminate signal at SM" << my_rank << Postoffice::Get()->fchannel(my_channel) << ". Shut down.");
            break;
          }

          // send out sync requests to other nodes
          startSync();

          // sync report
          time_since_last_report.stop();
          if (time_since_last_report.elapsed_ms() >= 10000) { // report every 10s
            auto ticks = server->getTotalTicks();
            ALOG("SM" << my_rank << Postoffice::Get()->fchannel(my_channel) <<
                 " report at sync " << syncs <<": " << std::setprecision(5) <<
                 1000.0*syncs_since_last_report/time_since_last_report.elapsed_ms() << " syncs/s. " <<
                 1000.0*(ticks-ticks_at_last_report)/Postoffice::Get()->num_worker_threads()/time_since_last_report.elapsed_ms() << " ticks/s avg. " <<
                 "noc estimate avg: " << action_timer.get_avg_estimate());
            updated_since_last_report = 0;
            total_since_last_report = 0;
            syncs_since_last_report = 0;
            ticks_at_last_report = ticks;
            time_since_last_report.start();
          } else {
            time_since_last_report.resume();
          }

          // pause between two synchronization rounds
          wait_function_();
        }
      }

      sw_runtime.stop();
      sw_pausing.stop();
      auto total = sw_pausing.elapsed_ms() + sw_finishing.elapsed_ms() + sw_collecting.elapsed_ms();
      ALOG("SM" << std::setprecision(4) << my_rank << Postoffice::Get()->fchannel(my_channel) << " terminates. " <<
           syncs << " syncs in " << sw_runtime.elapsed_s() << "s (" << 1.0*syncs/sw_runtime.elapsed_s() << " syncs/s)" <<
           " noc estimate: " << action_timer.get_avg_estimate() << ". \n" <<
           "Communicated " <<
           100.0*updated/total << "% of keys (threshold=" << sync_threshold << "). Sent " <<
           100.0 * numRefreshsNonZero / (numRefreshsRequested+0.000001) << "% of " <<
           numRefreshsRequested << " requested refreshes non-zero. \n" <<
           "Time spent: " << 100.0 * sw_pausing.elapsed_ms() / total << "% pausing, " <<
           100.0 * sw_finishing.elapsed_ms() / total << "% finishing, " <<
           100.0 * sw_collecting.elapsed_ms() / total << "% collecting.\n" <<
           "Repl/Rel " << num_replicate << "/" << num_relocate << ". " <<
           "Mean response hops: " << 1.0*total_refresh_hops / (num_refreshes_answered+0.000001) <<
           "\nHop stats: " << str(hop_stats));
    }

    void stop() {
      terminate = true;
    }

    // determine whether another node is working on a key
    bool intentOfOtherNode(const Key key, const int requester_rank) {
      for (int rank=0; rank != world_size; ++rank) {
        if (rank != requester_rank &&
            node_intent[rank].find(key) != node_intent[rank].end()) {
          return true;
        }
      }
      return false;
    }

    /**
     * \brief Process a SYNC message.
     *
     *        There is two types of such a message:
     *        1) "update" from the replica holder to the parameter owner (request)
     *        2) "refresh" from the parameter owner to the replica holder
     */
    void ProcessSyncMessage(const Message& msg, const bool forward) {
      if (terminate) { return; } // don't process messages after we received a terminate;
      KVPairs<Val> kvs;
      kvs.keys = msg.data[0];
      kvs.vals = msg.data[1];

      size_t len;
      size_t pos = 0;

      if (msg.meta.request) { // the parameter owner received a "update" message from the replica holder
        auto includes_update  = msg.GetBitset(msg.data[2]);
        auto refresh_desired   = msg.GetBitset(msg.data[3]);
        SArray<Version> refresh_versions (msg.data[4]);
        SArray<Key> inform_relocations (msg.data[5]);
        SArray<Key> relocation_counters (msg.data[6]);
        assert(forward || inform_relocations.size() == 0); // inform_relocations used only in forwards

        size_t senderRank = Postoffice::Get()->IDtoRank(msg.meta.sender);

        // inform this node about relocations (empty at first hop)
        assert(inform_relocations.size() == relocation_counters.size());
        for (size_t i=0; i!=inform_relocations.size(); ++i) {
          Key key = inform_relocations[i];
          server->addressbook.updateResidence(key, senderRank, relocation_counters[i]);
        }

        // clear intent of this node (will add keys of this message below)
        if (!forward) node_intent[senderRank].clear();

        // pre-allocate memory for response (i.e., refresh)
        auto total_len = handle->get_total_len(kvs.keys);
        SyncData<Val> response;
        response.reserve(kvs.keys.size(), total_len);

        std::vector<SyncData<Val>> forwards (Postoffice::Get()->num_servers());
        for(auto& forward : forwards) {
          // assume max. 1/5 of parameters needs forwarding
          forward.reserve(kvs.keys.size() * 0.2 / forwards.size(),
                                total_len * 0.2 / forwards.size());
        }

        int num_refreshed_keys = 0;
        auto worker_clocks = server->getWorkerClocks();

        handle->lockAll();

        // go through keys
        for (size_t i = 0, i_versions=0; i!=kvs.keys.size(); ++i) {
          Key key = kvs.keys[i];
          size_t len = handle->get_len(key);
          bool update = includes_update[i];
          bool refresh = refresh_desired[i];
          Version ver = 0;

          Val* val_update = nullptr;
          if (update) {
            val_update = &kvs.vals[pos];
            pos += len;
            assert(kvs.vals.size() >= pos);
          }

          if (refresh) {
            ver = refresh_versions[i_versions];
            ++i_versions;
          }

          handle->lockSingle(key);

          if (handle->isOwnerUnsafe(key)) {
            // The node we are on is the owner of the key. I.e., process the sync request

            bool relocate = false;
            if (refresh) {
              ++numRefreshsRequested; // stats

              bool localIntent = handle->hasLocalIntentUnsafe(key, worker_clocks);
              bool other_intent =
                (localIntent) || // intent of the node that we are currently on (the owner of the key)
                intentOfOtherNode(key, senderRank); // intent of other nodes

              if (Postoffice::Get()->management_techniques() == MgmtTechniques::REPLICATION_ONLY) {
                // only replication is possible -> replicate
                ++num_replicate;
                relocate = false;
              } else if (Postoffice::Get()->management_techniques() == MgmtTechniques::RELOCATION_ONLY) {
                // only relocation is possible -> relocate
                ++num_relocate;
                relocate = true;
              } else {
                // both replication and replication are possible
                if (other_intent) { // -> replicate if there is intent by another node
                  ++num_replicate;
                  relocate = false;
                } else { // -> relocate if there is no other intent
                  ++num_relocate;
                  relocate = true;
                }
              }

              // the node has intent for this key (we use this to make relocate/replicate decisions)
              node_intent[senderRank].insert(key);
            }

            bool copied_new_version;
            Version latest_main_ver;
            Version relocation_counter;
            std::tie(copied_new_version, latest_main_ver, relocation_counter) =
              handle->writeCopyDropOwnedKeyUnsafe(key, val_update, refresh, response.kvs, ver, relocate);

            if (refresh) {
              ++num_refreshed_keys;
              if (copied_new_version || relocate) { // include the key in the response if we have an update or relocate it
                response.kvs.keys.push_back(key);
                response.relocate.push_back(relocate);
                response.includes_update.push_back(copied_new_version);

                if (copied_new_version) { // we found updates that are new for the refresh requester
                  response.versions.push_back(latest_main_ver);
                  ++numRefreshsNonZero;
                }

                if (relocate) {
                  // increase the relocation counter
                  ++relocation_counter;

                  // transfer the relocation counter to the new owner
                  response.relocation_counters.push_back(relocation_counter);

                  // inform home node about relocation
                  auto home_node = server->addressbook.getManager(key);
                  if (my_rank == home_node) { // the current node is the key's home node
                    server->addressbook.updateResidenceUnsafe(key, senderRank, relocation_counter);
                  } else {
                    // inform the home node about the relocation (unless the target is the home node)
                    if (senderRank != home_node) {
                      forwards[server->addressbook.getManager(key)].inform_relocations.push_back(key);
                      forwards[server->addressbook.getManager(key)].relocation_counters.push_back(relocation_counter);
                    }

                    // store the relocation target in the location cache
                    if (Postoffice::Get()->use_location_caches()) {
                      server->addressbook.updateLocationCacheUnsafe(key, senderRank);
                    }
                  }
                }
              }
            }
          } else {
            // The node we are on is not the owner of this key. I.e., forward the sync request
            auto destination = server->addressbook.getDirectionsUnsafe(key, msg.meta.hops <= 2); // use location caches in the first 2 hops and the home node afterwards
            forwards[destination].kvs.keys.push_back(key);
            forwards[destination].includes_update.push_back(update);
            forwards[destination].refresh_desired.push_back(refresh);
            if (update) {
              std::copy_n(val_update, handle->get_len(key), std::back_inserter(forwards[destination].kvs.vals));
            }
            if (refresh) {
              forwards[destination].versions.push_back(ver);
            }
          }

          handle->unlockSingle(key);
        }
        handle->unlockAll();

        // respond to the requester
        assert(response.kvs.keys.size() == response.relocate.size());
        assert(response.kvs.keys.size() == response.includes_update.size());
        assert(response.includes_update.count() == response.versions.size());

        if (num_refreshed_keys != 0) { // always: `num_refreshed_keys` >= kvs_response.keys.size()
          Message res;
          res.meta.app_id = msg.meta.app_id;
          res.meta.hops = msg.meta.hops + 1;
          res.meta.channel = my_channel;
          res.meta.customer_id = 0;
          res.meta.request     = false;
          res.meta.head        = Control::SYNC;
          res.meta.recver      = msg.meta.sender;
          res.meta.timestamp   = num_refreshed_keys; // NOTE: we abuse the `timestamp` field here
          res.AddData(response.kvs.keys);
          res.AddData(response.kvs.vals);
          res.AddData(response.includes_update);
          res.AddData(response.relocate);
          res.AddData(response.versions);
          res.AddData(response.relocation_counters);
          Postoffice::Get()->van()->Send(res);
        }

        // send messages to other nodes (sync requests and home node relocation info)
        for (size_t rank=0; rank!=forwards.size(); ++rank) {
          if (forwards[rank].kvs.keys.size() != 0 || forwards[rank].inform_relocations.size() != 0) {

            sendSyncRequest(rank, forwards[rank], msg.meta.app_id, msg.meta.hops+1, true, msg.meta.sender);
          }
        }
      } else { // the replica holder received a "refresh" message from the parameter owner

        // parse the response
        auto includes_update    = msg.GetBitset(msg.data[2]);
        auto relocate = msg.GetBitset(msg.data[3]);
        SArray<Version> versions (msg.data[4]);
        SArray<Version> relocation_counters (msg.data[5]);
        auto senderRank = Postoffice::Get()->IDtoRank(msg.meta.sender);

        assert(kvs.keys.size() == relocate.size());
        assert(kvs.keys.size() == includes_update.size());
        assert(includes_update.count() == versions.size());
        assert(relocate.count() == relocation_counters.size());

        // process replicas: write updates and/or upgrade the replica to an owned key
        for (size_t i = 0, i_update = 0, i_relocate = 0; i < kvs.keys.size(); ++i) {
          Key key = kvs.keys[i];
          len = handle->get_len(key);

          Val* update = nullptr;
          Version version = -1;
          if (includes_update[i]) {
            update = &kvs.vals[pos];
            pos += len;
            version = versions[i_update];
            ++i_update;
          }

          Version relocation_counter = 0;
          if (relocate[i]) {
            relocation_counter = relocation_counters[i_relocate];
            ++i_relocate;
          }

          auto home_node = server->addressbook.getManager(key);

          handle->lockSingle(key); // acquire lock for key --------------------------------------------------

          handle->refreshUpgradeReplicaUnsafe(key, update, version, relocate[i], relocation_counter);

          // update residence information in case we just moved the parameter to the home node
          // (we don't send an extra message in this case)
          if (relocate[i] && my_rank == home_node) {
            server->addressbook.updateResidenceUnsafe(key, my_rank, relocation_counter);
          }

          // update the location cache from this sync response
          if (!relocate[i] && Postoffice::Get()->use_location_caches() && my_rank != home_node) {
            server->addressbook.updateLocationCacheUnsafe(key, senderRank);
          }

          handle->unlockSingle(key); // give up lock --------------------------------------------------------
        }

        // stats
        ++num_responses;
        num_refreshes_answered += msg.meta.timestamp;
        total_refresh_hops += msg.meta.timestamp * msg.meta.hops;
        hop_stats[std::min(static_cast<size_t>(msg.meta.hops), hop_stats.size()-1)] += msg.meta.timestamp;
      }
    }

    /**
     * \brief Add program options for parameter replication
     */
    static void AddSyncOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("sys.sync.max_per_sec", po::value<double>(&syncs_per_sec)->default_value(1000), "maximum number of synchronization rounds per second. Set 0 to set no limit.")
        ("sys.sync.pause", po::value<int>(&sync_pause)->default_value(0), "pause between two background synchronization runs (in milliseconds). You can seit either this pause OR a max. number of syncs per sec.")
        ("sys.sync.threshold", po::value<double>(&sync_threshold)->default_value(0), "synchronize only updates larger than a threshold. Options: -1 (sync all updates, including zero ones), 0 (sync all non-zero updates [default]), >0 (sync all updates where l2(update)>=threshold, inf (disable sync entirely)")
        ;

        ActionTimer::AddProgramOptions(options);
    }

    /**
     * \brief Print sync options
     */
    static std::string PrintOptions() {
      std::stringstream s;
      s << "Sync options: nw threads " << Postoffice::Get()->get_num_network_threads() << ", sync threshold " << sync_threshold << ", pause " << sync_pause << ", max sps " << syncs_per_sec;
      return s.str();
    }
  };

  // static members for program options. repeat default values in case `AddSyncOptions()` is not called.
  template <typename Val, typename Handle> int SyncManager<Val,Handle>::sync_pause = 0;
  template <typename Val, typename Handle> double SyncManager<Val,Handle>::syncs_per_sec = 1000;
  template <typename Val, typename Handle> double SyncManager<Val,Handle>::sync_threshold = 0;
}


#endif  // PS_SYNC_MANAGER_H_

