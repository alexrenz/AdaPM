/**
 *  Copyright (c) 2020 Alexander Renz-Wieland
 */
#ifndef PS_REPLICA_MANAGER_H_
#define PS_REPLICA_MANAGER_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <valarray>
#include <unordered_map>
#include <ps/internal/postoffice.h>
#include <zmq_van.h>
#include "../apps/utils.h"
#include "ps/kv_app.h"
#include <math.h>
#include <boost/program_options.hpp>


#include <iostream>
#include <sstream>

const int REP_MGRS_START_PORT = 52000;

// replica synchronization methods
enum ReplicaSyncMethod {BG_TREE, BG_BUTTERFLY};
std::ostream& operator<<(std::ostream &o, const ReplicaSyncMethod& rsm) {
  switch(rsm) {
  case BG_TREE: return o << "bg_tree";
  case BG_BUTTERFLY: return o << "bg_butterfly";
  default: return o << "unkown";
  }
}
std::istream& operator>>(std::istream& in, ReplicaSyncMethod& rsm) {
  std::string token; in >> token;
  if (token == "bg_tree") rsm = BG_TREE;
  else if (token == "bg_butterfly") rsm = BG_BUTTERFLY;
  else { CHECK(false) << "Fatal! Unkown replica synchronization method " << token; }
  return in;
}

namespace ps {
  const Key SHUTDOWN = -1;




  template <typename Val, typename Handle>
  class ReplicaManager {

    // synchronization method
    static ReplicaSyncMethod rsm;

    // threshold for selective synchronization (see program options for description)
    static double sync_threshold;

    // whether to use a separate ZMQ context for replica synchronization communication
    static bool use_separate_zmq_context;

    // holds sockets to other replica managers
    std::unordered_map<int, void*> sockets;

    // rank of this replica manager
    int my_rank;

    // total number of replica managers
    int world_size;

    // reference to the parameter store
    Handle& handle;

    // counter for synchronization rounds
    std::atomic<unsigned long> syncs {0};
    unsigned long syncs_since_last_report = 0;

    // replica manager thread terminates as soon as this is set to true
    bool terminate = false;

    // average updates (true) or sum them (false, default)
    static bool average_updates;

    // update clipping
    static double clip_factor;
    bool init_phase = false;

    // average norm
    double norm_sum = 0;
    size_t norm_num = 0;
    const size_t norm_num_warmup = 1000;

    // replica synchronization function
    std::function<void(KVPairs<Val>& kvs)> synchronization_function_;

    // waiting between synchronization rounds
    std::function<void()> wait_function_; // wait function (multiple options available)
    static int sync_pause; // period pause (in milliseconds)
    static double syncs_per_sec; // aim for specific number of syncs per seconds
    std::chrono::milliseconds sync_interval {}; // interval (calculated, for interval pauses)
    chrono::time_point<std::chrono::high_resolution_clock> last_run; // last run (for interval pauses)

    // stopwatches
    util::Stopwatch sw_runtime, sw_collect, sw_write, sw_exchange, time_since_last_report;

    // stats
    long updated = 0;
    long total = 0;
    long updated_since_last_report = 0;
    long total_since_last_report = 0;
    size_t num_clips = 0;
    size_t num_updates = 0;
    size_t sent_bytes_total = 0;
    size_t received_bytes_total = 0;

    // currently synchronized (global) state of replicas
    KVPairs<Val> replicas;

    // ZMQ context (either the same as push/pull/localize or a separate one)
    void* context_;

    /**
     * \brief Synchronize updates among nodes
     */
    int sync (bool shutdown = false) {

      KVPairs<Val> updates;

      // collect local updates
      if (shutdown) {
        updates.keys.resize(1);
        updates.keys[0] = SHUTDOWN;
      } else {
        sw_collect.resume();
        handle.readReplicas(replicas.keys, updates, sync_threshold);
        total += replicas.keys.size();
        updated += updates.keys.size();
        total_since_last_report += replicas.keys.size();
        updated_since_last_report += updates.keys.size();
        sw_collect.stop();
      }

      // synchronize updates among all replica managers
      sw_exchange.resume();
      synchronization_function_(updates);
      sw_exchange.stop();

      // propagated shutdown signal
      if (updates.keys.size() == 1 && updates.keys[0] == SHUTDOWN)
        return -1;

      // update replica state and local parameter storage
      sw_write.resume();
      size_t replicas_pos = 0;
      size_t updates_pos = 0;
      for (size_t i=0, j=0; i!=updates.keys.size(); ++i) {
        Key key = updates.keys[i];
        auto len = handle.get_len(key);
        ++num_updates;

        // move to correct position in state
        while (replicas.keys[j] != key) {
          replicas_pos += handle.get_len(replicas.keys[j]);
          ++j;
        }

        double update_factor = (average_updates && !init_phase) ? 1.0 / world_size : 1;

        // calculate update vector norm
        double norm = 0;
        for(size_t k=0; k!=len; ++k) {
          norm += updates.vals[updates_pos+k]*updates.vals[updates_pos+k];
        }
        norm = sqrt(norm);

        // record the average norm
        if(!init_phase) { // do not count model initialization updates
          norm_sum += norm;
          norm_num += 1;
        }

        // clip replica updates if (1) clipping is enabled,
        // (2) we are not in an init phase and (3) we passed the warmup
        if (clip_factor != 0 && !init_phase && norm_num > norm_num_warmup) {
          double avg_norm = 1.0 * norm_sum / norm_num;
          double threshold = clip_factor * avg_norm;
          if (norm > threshold) {
            update_factor = update_factor / norm * threshold;
            ++num_clips;
          }
        }

        // update replicas
        for(size_t k=0; k!=len; ++k) {
          replicas.vals[replicas_pos+k] += updates.vals[updates_pos+k] * update_factor;
        }

        // write updated state to the local parameter storage
        handle.writeReplica(key, &replicas.vals.data()[replicas_pos]);
        updates_pos += len;
      }
      sw_write.stop();

      ++syncs;
      ++syncs_since_last_report;

      return 0;
    }

    /**
     * \brief Send a message to another replica manager
     */
    int SendMessage(const int rank_to, KVPairs<Val>& kvs) {
      // find socket
      assert(sockets.find(rank_to) != sockets.end());
      void *socket = sockets[rank_to];

      int tag = ZMQ_SNDMORE;
      int n = 2; // send one message for keys and one for values
      int send_bytes = 0;

      // send data
      for (int i = 0; i < n; ++i) {
        zmq_msg_t data_msg;
        SArray<char>* data = i==0 ? new SArray<char>(kvs.keys) : new SArray<char>(kvs.vals);
        int data_size = data->size();
        zmq_msg_init_data(&data_msg, data->data(), data->size(), FreeData, data);
        if (i == n - 1) tag = 0;
        while (true) {
          if (zmq_msg_send(&data_msg, socket, tag) == data_size) break;
          if (errno == EINTR) continue;
          LOG(WARNING) << "failed to send message to node [" << rank_to
                       << "] errno: " << errno << " " << zmq_strerror(errno)
                       << ". " << i << "/" << n;
          return -1;
        }
        // zmq_msg_close(&data_msg);
        send_bytes += data_size;
      }
      sent_bytes_total += send_bytes;
      return send_bytes;
    }


    /**
     * \brief Receive a message from another replica manager
     */
    int ReceiveMessage(const int rank_from, KVPairs<Val>& kvs) {
      // find socket
      assert(sockets.find(rank_from) != sockets.end());
      void *socket = sockets[rank_from];

      // receive and process message parts
      size_t recv_bytes = 0;
      for (int i = 0; ; ++i) {
        zmq_msg_t* zmsg = new zmq_msg_t;
        CHECK(zmq_msg_init(zmsg) == 0) << zmq_strerror(errno);
        while (true) {
          if (zmq_msg_recv(zmsg, socket, 0) != -1) break;
          if (errno == EINTR) {
            std::cout << "interrupted";
            continue;
          }
          LOG(WARNING) << "failed to receive message. errno: "
                       << errno << " " << zmq_strerror(errno);
          return -1;
        }
        char* buf = CHECK_NOTNULL((char *)zmq_msg_data(zmsg));
        size_t size = zmq_msg_size(zmsg);
        recv_bytes += size;

        // store received data in given key-value object
        SArray<char> data;
        data.reset(buf, size, [zmsg, size](char* buf) {
            zmq_msg_close(zmsg);
            delete zmsg;
          });
        if (i == 0) {
          kvs.keys = data;
        } else {
          kvs.vals = data;
        }
        if (!zmq_msg_more(zmsg)) { break; }
      }
      received_bytes_total += recv_bytes;
      return recv_bytes;
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
      if (syncs != 0) { // (don't sync on first call)
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

    /**
     * \brief Opens connections to all other replica managers
    */
    void init () {

      Van* van = Postoffice::Get()->van();

      // create new or get existing ZMQ context
      if (use_separate_zmq_context) {
        // create a new zmq context (independent of the context for push/pull/localizes)
        context_ = zmq_ctx_new();
        CHECK(context_ != NULL) << "create replica manager 0mq context failed";
        zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 65536);
        zmq_ctx_set(context_, ZMQ_IO_THREADS, 1);
      } else {
        // use the same zmq context as push/pull/localize
        context_ = static_cast<ZMQVan*>(van)->getContext();
      }

      // connect all replica managers
      auto& connected_nodes = van->GetConnectedNodes();
      for (auto node : connected_nodes) {
        auto rank = Postoffice::Get()->IDtoRank(node.second);
        if (node.second == 1) {
          // no connection to scheduler
        } else if (rank == my_rank) {
          // no connection to the same node
        } else if (rank > my_rank) {
          // bind
          auto port = REP_MGRS_START_PORT + rank*100 + my_rank;
          std::string addr = "tcp://*:" + std::to_string(port);
          // ADLOG("Rank " << my_rank << " binds to rank " << rank << ", address: " << addr);

          sockets[rank] = zmq_socket(context_, ZMQ_PAIR);
          int rc = zmq_bind (sockets[rank], addr.c_str());
          if(rc != 0) ADLOG("Failed to bind on rank " << my_rank << " (to rank " << rank << "): " << addr);
          assert (rc == 0);
        } else {
          // connect
          auto port = REP_MGRS_START_PORT + my_rank*100 + rank;
          std::string ip = node.first.substr(0,node.first.find(":"));
          std::string addr = "tcp://" + ip + ":" + std::to_string(port);
          // ADLOG("Rank " << my_rank << " connects to rank " << rank << ", address: " + addr);

          sockets[rank] = zmq_socket(context_, ZMQ_PAIR);
          int rc = zmq_connect (sockets[rank], addr.c_str());
          if(rc != 0) ADLOG("Failed to connect on rank " << my_rank << " (to rank " << rank << "): " << addr);
          assert (rc == 0);
        }
      }
    }

    /**
     * \brief Merge two KVPairs objects
     */
    KVPairs<Val> merge(KVPairs<Val>& a, KVPairs<Val>& b) {
      KVPairs<Val> target;

      // propagate the shutdown signal
      if ((a.keys.size() != 0 && a.keys[0] == SHUTDOWN) || (b.keys.size() != 0 && b.keys[0] == SHUTDOWN)) {
        ADLOG("Propagate shutdown signal.");
        target.keys.resize(1);
        target.keys[0] = SHUTDOWN;
        return target;
      }

      size_t i = 0, j = 0;
      size_t i_pos = 0, j_pos = 0;

      // pre-allocate (for efficiency)
      auto max_keys = min(replicas.keys.size(), a.keys.size()+b.keys.size());
      target.keys.reserve(max_keys);
      target.vals.reserve(replicas.vals.size());

      // Merge a and b
      while (i != a.keys.size() && j != b.keys.size()) {
        if (a.keys[i] == b.keys[j]) {
          auto len = handle.get_len(a.keys[i]);
          // merge a parameter from a and b
          target.keys.push_back(a.keys[i]);
          size_t start = target.vals.size();
          target.vals.resize(target.vals.size()+len);
          for (size_t c = 0; c != len; ++c) {
            target.vals[start+c] = a.vals[i_pos+c] + b.vals[j_pos+c];
          }
          ++i;
          ++j;
          i_pos += len;
          j_pos += len;
        } else if (a.keys[i] < b.keys[j]) {
          // copy a parameter from a
          Key key = a.keys[i];
          auto len = handle.get_len(key);
          target.keys.push_back(key);
          std::copy_n(a.vals.begin()+i_pos, len, std::back_inserter(target.vals));
          ++i;
          i_pos += len;
        } else {
          // copy a parameter from b
          Key key = b.keys[j];
          auto len = handle.get_len(key);
          target.keys.push_back(key);
          std::copy_n(b.vals.begin()+j_pos, len, std::back_inserter(target.vals));
          ++j;
          j_pos += len;
        }
      }

      // Copy remaining parameters in a
      while (i != a.keys.size()) {
        Key key = a.keys[i];
        auto len = handle.get_len(key);
        target.keys.push_back(key);
        std::copy_n(a.vals.begin()+i_pos, len, std::back_inserter(target.vals));
        ++i;
        i_pos += len;
      }

      // Copy remaining parameters in b
      while (j != b.keys.size()) {
        Key key = b.keys[j];
        auto len = handle.get_len(key);
        target.keys.push_back(key);
        std::copy_n(b.vals.begin()+j_pos, len, std::back_inserter(target.vals));
        ++j;
        j_pos += len;
      }

      return target;
    }

    /**
     * \brief Exchange parameter updates among all nodes using Butterfly AllReduce
     */
    void butterfly (KVPairs<Val>& kvs) {
      size_t rounds = log2(world_size);


      for (size_t round = 0; round!=rounds; ++round) {

        // determine exchange partner
        auto group_size = pow(2,round+1);
        auto group_offset = floor(my_rank / group_size) * group_size;
        auto rank_in_group = my_rank - group_offset;
        auto to = group_offset + group_size-1 - rank_in_group;
        // ADLOG("Butterfly round " << round << ": rank " << my_rank << " sends to rank " << to << ": keys " << kvs.keys << " vals " << kvs.vals);

        // send, receive, and merge update
        KVPairs<Val> outgoing;
        outgoing.keys.CopyFrom(kvs.keys);
        outgoing.vals.CopyFrom(kvs.vals);
        SendMessage(to, outgoing);
        KVPairs<Val> incoming;
        ReceiveMessage(to, incoming);
        kvs = merge(outgoing, incoming); // copies
      }
    }


    /**
     * \brief Exchange parameter updates among all nodes using Tree AllReduce
     */
    void tree (KVPairs<Val>& kvs) {
      size_t rounds = log2(world_size); // twice


      // gather
      for (size_t round = 0; round!=rounds; ++round) {
        // what does this node do in this round?
        if (my_rank % int(pow(2,round+1)) == 0) {
          // receive update an merge
          auto partner = my_rank + pow(2,round);
          /* ADLOG("Gather round " << round << ": rank " << my_rank << " receives from rank " << partner); */
          KVPairs<Val> incoming;
          ReceiveMessage(partner, incoming);
          kvs = merge(kvs, incoming);
        } else if ( my_rank % int(pow(2,round+1)) == int(pow(2,round)) ) {
          // send update
          auto partner = my_rank - pow(2,round);
          /* ADLOG("Gather round " << round << ": rank " << my_rank << " sends to rank " << partner); */
          SendMessage(partner, kvs);
        } else {
          // do nothing in this round
        }
      }

      // scatter
      for (int round = rounds-1; round!=-1; --round) {
        // what does this node do in this round?
        if (my_rank % int(pow(2,round+1)) == 0) {
          // send out update
          auto partner = my_rank + pow(2,round);
          /* ADLOG("Scatter round " << round << ": rank " << my_rank << " sends to rank " << partner); */
          SendMessage(partner, kvs);
        } else if ( my_rank % int(pow(2,round+1)) == int(pow(2,round)) ) {
          // receive update
          auto partner = my_rank - pow(2,round);
          /* ADLOG("Scatter round " << round << ": rank " << my_rank << " receives from rank " << partner); */
          ReceiveMessage(partner, kvs);
        } else {
          // do nothing in this round
        }
      }
    }

  public:
    /**
     * \brief Construct replica manager object
     */
    ReplicaManager(Handle& h, vector<Key>* replicated_parameters) : handle(h), replicas{} {
      my_rank = Postoffice::Get()->my_rank();
      world_size = Postoffice::Get()->num_servers();

      // initialize state of replicas
      if (replicated_parameters != nullptr) {
        replicas.keys = SArray<Key>(*replicated_parameters);
        std::sort(replicas.keys.begin(), replicas.keys.end()); // we rely on replica keys to be sorted (in merge fn)
      }
      replicas.vals.resize(handle.get_total_len(replicas.keys));
      std::fill(replicas.vals.begin(), replicas.vals.end(), 0);

      // initialize replication
      handle.initializeReplication(replicas.keys);

      // set replica sync method
      if (rsm == ReplicaSyncMethod::BG_TREE) {
        if (replicas.keys.size() != 0) assert ((world_size & (world_size - 1)) == 0);
        synchronization_function_ = std::bind(&ReplicaManager::tree, this, std::placeholders::_1);
      } else if (rsm == ReplicaSyncMethod::BG_BUTTERFLY) {
        if (replicas.keys.size() != 0) assert ((world_size & (world_size - 1)) == 0);
        synchronization_function_ = std::bind(&ReplicaManager::butterfly, this, std::placeholders::_1);
      } else {
        ADLOG("Unknown replica synchronization method: " << rsm);
      }

      // set wait method
      assert(sync_pause == 0 || syncs_per_sec == 0);
      if (sync_pause == 0 && syncs_per_sec == 0)
        wait_function_ = std::bind(&ReplicaManager::wait_none, this);
      else if (sync_pause != 0)
        wait_function_ = std::bind(&ReplicaManager::wait_period, this);
      else {
        assert(syncs_per_sec >= 0);
        sync_interval = std::chrono::milliseconds(static_cast<size_t>(1000 / syncs_per_sec));
        wait_function_ = std::bind(&ReplicaManager::wait_interval, this);
      }
    }

    /**
     * \brief Wait until all updates that have reached the store by now
     *        are propagated to all replicas
     */
    void WaitReplicaSync() {
      // no need to wait if there is only 1 node or no replicas
      if (world_size == 1 || replicas.keys.size() == 0) {
        return;
      }

      // no need wait if sync is disabled entirely
      if (std::isinf(sync_threshold)) {
        return;
      }

      // wait to syncs into the future as the current one can be running already
      unsigned long current_sync = syncs;
      unsigned long wait_sync = current_sync + 2;

      while (syncs < wait_sync) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }

      ADLOG("Waited for replica sync " << wait_sync << " (called at sync " << current_sync << ")");
    }

    /**
     * \brief Separate Replica Manager thread that synchronizes nodes in the background
     */
    void thread () {
      // do nothing if there is only one process
      if (world_size == 1) {
        ADLOG("Replica Manager: don't start as there is only one process globally.");
        return;
      }

      // initialize replica manager
      init();

      ADLOG("Replica Manager " << Postoffice::Get()->my_rank() << " set up (" << replicas.keys.size() << " parameters, " << std::setprecision(5) << sizeof(Val) * replicas.vals.size() / 1024.0 / 1024 << " MB). Starting sync (" << rsm << ").");
      sw_runtime.start();
      time_since_last_report.start();

      // initial wait (for setup mostly)
      wait_function_();

      // replica manager background loop
      if (!std::isinf(sync_threshold)) { // threshold=inf disables sync entirely
        while (true) {
          if (terminate) {
            // graceful shutdown of replica managers
            ADLOG("Terminate signal at replica manager rank " << my_rank << ". Shut down all replica managers.");
            // Can stop directly if sync is disabled. Otherwise, propagate the termination signal to make sure all
            // replica managers cleanly exit in the same sync round (this one).
          }

          // synchronize replicas
          auto rc = sync(terminate);
          if (rc == -1) break; // this was the last round. exit

          // debug output: report share of keys communicated so far
          if (syncs % 200 == 0) {
            time_since_last_report.stop();
            ADLOG("RM " << my_rank << " report at sync " << syncs <<": " << std::setprecision(5) <<
                  1.0*syncs_since_last_report/time_since_last_report.elapsed_s() <<
                  " syncs/s and " <<
                  100.0 * updated_since_last_report / total_since_last_report <<
                  "% selectivity since last report (" <<
                  100.0 * updated / total << "% overall). " << 100.0 * num_clips / num_updates << "% of " << num_updates << " updates clipped, average norm " << norm_sum / norm_num);
            updated_since_last_report = 0;
            total_since_last_report = 0;
            syncs_since_last_report = 0;
            time_since_last_report.start();
          }

          // pause between two synchronization rounds
          wait_function_();
        }
      }

      sw_runtime.stop();
      ADLOG("Replica Manager of rank " << std::setprecision(4) << my_rank << " terminates. " << syncs << " syncs in " << sw_runtime.elapsed_s() << "s (" << 1.0*syncs/sw_runtime.elapsed_s() << " syncs/s)\n" << "Timings (ms/sync). " << 1.0/1000*sw_collect.elapsed_us()/syncs << " collect, " << 1.0/1000*sw_exchange.elapsed_us()/syncs << " exchange, " << 1.0/1000*sw_write.elapsed_us()/syncs << " write" << "\nCommunicated " << 100.0*updated/total << "% of keys (threshold=" << sync_threshold << "). Sent " << sent_bytes_total/1024/1024/sw_runtime.elapsed_s() << " and received " << received_bytes_total/1024/1024/sw_runtime.elapsed_s() << " MB/s on average");

      // send a closing message to all open sockets
      KVPairs<Val> close;
      for (auto socket : sockets) {
        SendMessage(socket.first, close);
      }

      // shut down each socket after receiving the closing message
      for (auto socket : sockets) {
        ReceiveMessage(socket.first, close);
        CHECK_EQ(zmq_close(socket.second), 0);
      }

      // close context (if we created a new one)
      if (use_separate_zmq_context) {
        zmq_ctx_destroy(context_);
        context_ = nullptr;
      }
    }

    void stop() {
      terminate = true;
    }


    /**
     * \brief Enable or disable the init phase.
     *        In the init phase, all updates are written 1:1, i.e., no update averaging or clipping.
     *        We use this for initializing models
     */
    void setInit (const bool init) {
      init_phase = init;
      ADLOG("Replica update aggregation: " << (init_phase ? "init" : "regular"));
    }

    /**
     * \brief Add program options for parameter replication
     */
    static void AddReplicationOptions(boost::program_options::options_description& options) {
      namespace po = boost::program_options;
      options.add_options()
        ("rep.sm", po::value<ReplicaSyncMethod>(&rsm)->default_value(ReplicaSyncMethod::BG_BUTTERFLY), "replica synchronization method")
        ("rep.pause", po::value<int>(&sync_pause)->default_value(0), "pause between two background synchronization runs (in milliseconds)")
        ("rep.syncs_per_sec", po::value<double>(&syncs_per_sec)->default_value(5), "number of synchronization rounds per second (goal) (default: 5 per second)")
        ("rep.threshold", po::value<double>(&sync_threshold)->default_value(0), "synchronize only updates larger than a threshold. Options: -1 (sync all updates, including zero ones), 0 (sync all non-zero updates [default]), >0 (sync all updates where l2(update)>=threshold, inf (disable sync entirely)")
        ("rep.separate_zmq_context", po::value<bool>(&use_separate_zmq_context)->default_value(false), "use a separate ZMQ context for replica synchronization (i.e., not the same as for push/pull/localize)")
        ("rep.average_updates", po::value<bool>(&average_updates)->default_value(false), "average the updates of replicas")
        ("rep.clip_updates", po::value<double>(&clip_factor)->default_value(0), "clip replica updates (default: 0, i.e., no clipping)")
        ;
    }

    /**
     * \brief Print replication options
     */
    static std::string PrintOptions() {
      std::stringstream s;
      s << "Replication options: rsm " << rsm << ", nw threads " << Postoffice::Get()->get_num_network_threads() << ", separate context " << use_separate_zmq_context << ", sync threshold " << sync_threshold << ", pause " << sync_pause << ", sps " << syncs_per_sec << ", average updates " << average_updates << ", clip factor " << clip_factor;
      return s.str();
    }

    /**
     * \brief Return synchronization method
     */
    static ReplicaSyncMethod const get_rsm() {
      return rsm;
    }
  };

  // define static class members
  template <typename Val, typename Handle> int ReplicaManager<Val,Handle>::sync_pause;
  template <typename Val, typename Handle> double ReplicaManager<Val,Handle>::syncs_per_sec;
  template <typename Val, typename Handle> ReplicaSyncMethod ReplicaManager<Val,Handle>::rsm;
  template <typename Val, typename Handle> double ReplicaManager<Val,Handle>::sync_threshold;
  template <typename Val, typename Handle> bool ReplicaManager<Val,Handle>::use_separate_zmq_context;
  template <typename Val, typename Handle> bool ReplicaManager<Val,Handle>::average_updates;
  template <typename Val, typename Handle> double ReplicaManager<Val,Handle>::clip_factor;
}


#endif  // PS_REPLICA_MANAGER_H_

