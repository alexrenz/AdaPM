/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_POSTOFFICE_H_
#define PS_INTERNAL_POSTOFFICE_H_
#include <mutex>
#include <algorithm>
#include <vector>
#include <limits>
#include "ps/range.h"
#include "ps/internal/env.h"
#include "ps/internal/customer.h"
#include "ps/internal/van.h"
namespace ps {
/**
 * \brief the center of the system
 */
class Postoffice {
  template <typename Val, typename Handle>
  friend class ColoKVServer;
 public:
  /**
   * \brief return the singleton object
   */
  static Postoffice* Get() {
    static Postoffice e; return &e;
  }
  /** \brief get the van */
  Van* van() { return van_; }
  /**
   * \brief start the system
   *
   * This function will block until every nodes are started.
   * \param argv0 the program name, used for logging.
   * \param do_barrier whether to block until every nodes are started.
   */
  void Start(int customer_id, const char* argv0, const bool do_barrier);
  /**
   * \brief terminate the system
   *
   * All nodes should call this function before exiting.
   * \param do_barrier whether to do block until every node is finalized, default true.
   */
  void Finalize(const int customer_id, const bool do_barrier = true);
  /**
   * \brief add an customer to the system. threadsafe
   */
  void AddCustomer(Customer* customer);
  /**
   * \brief remove a customer by given it's id. threasafe
   */
  void RemoveCustomer(Customer* customer);
  /**
   * \brief get the customer by id, threadsafe
   * \param app_id the application id
   * \param customer_id the customer id
   * \param timeout timeout in sec
   * \return return nullptr if doesn't exist and timeout
   */
  Customer* GetCustomer(int app_id, int customer_id, int timeout = 0) const;
  /**
   * \brief get the id of a node (group), threadsafe
   *
   * if it is a  node group, return the list of node ids in this
   * group. otherwise, return {node_id}
   */
  const std::vector<int>& GetNodeIDs(int node_id) const {
    const auto it = node_ids_.find(node_id);
    CHECK(it != node_ids_.cend()) << "node " << node_id << " doesn't exist";
    return it->second;
  }
  /**
   * \brief return the key ranges of all server nodes
   */
  const std::vector<Range>& GetServerKeyRanges();
  /**
   * \brief the template of a callback
   */
  using Callback = std::function<void()>;
  /**
   * \brief Register a callback to the system which is called after Finalize()
   *
   * The following codes are equal
   * \code {cpp}
   * RegisterExitCallback(cb);
   * Finalize();
   * \endcode
   *
   * \code {cpp}
   * Finalize();
   * cb();
   * \endcode
   * \param cb the callback function
   */
  void RegisterExitCallback(const Callback& cb) {
    exit_callback_ = cb;
  }
  /**
   * \brief convert from a worker rank into a node id
   * \param rank the worker rank
   */
  static inline int WorkerRankToID(int rank) {
    return rank * 2 + 9;
  }
  /**
   * \brief convert from a server rank into a node id
   * \param rank the server rank
   */
  static inline int ServerRankToID(int rank) {
    return rank * 2 + 8;
  }
  /**
   * \brief convert from a node id into a server or worker rank
   * \param id the node id
   */
  static inline int IDtoRank(int id) {
#ifdef _MSC_VER
#undef max
#endif
    return std::max((id - 8) / 2, 0);
  }
  /** \brief Returns the number of worker nodes */
  int num_workers() const { return num_workers_; }
  /** \brief Returns the number of server nodes */
  int num_servers() const { return num_servers_; }
  /** \brief Returns the rank of this node in its group
   *
   * Each worker will have a unique rank within [0, NumWorkers()). So are
   * servers. This function is available only after \ref Start has been called.
   */
  int my_rank() const { return IDtoRank(van_->my_node().id); }
  /** \brief Returns true if this node is a worker node */
  int is_worker() const { return is_worker_; }
  /** \brief Returns true if this node is a server node. */
  int is_server() const { return is_server_; }
  /** \brief Returns true if this node is a scheduler node. */
  int is_scheduler() const { return is_scheduler_; }
  /** \brief Returns the verbose level. */
  int verbose() const { return verbose_; }
  /** \brief Return whether this node is a recovery node */
  bool is_recovery() const { return van_->my_node().is_recovery; }

  // methods added for co-location and parameter movements. start [sysChange]
  void setup(const Key num_keys, const unsigned int num_threads) {
    num_keys_ = num_keys;
    num_worker_threads_ = num_threads;
    customers_.resize(num_threads + num_channels_ + 1); // +1 for sampling customer
  }
  /** \brief Return whether to relocate parameters (otherwise, the system will always replicate on intent) */
  inline MgmtTechniques management_techniques() const { return management_techniques_; }
  /** \brief Whether and with which target probability to time intent actions */
  inline double time_intent_actions() const { return time_intent_actions_; }
  /** \brief Returns whether location caches are used */
  inline bool use_location_caches() const { return location_caches_; }
  /** \brief Returns the number of communication channels */
  inline unsigned int num_channels() const { return num_channels_; }
  /** \brief Get the configured number of keys */
  inline Key num_keys() const { return num_keys_; }
  /** \brief Get the number of worker threads per server process */
  inline uint num_worker_threads() const { return num_worker_threads_; }
  // number of network threads
  int get_num_network_threads() const { return num_network_threads_; }
  // end [sysChange]


   // customer ids (N worker threads, C channels):
   //  0..(N-1): worker threads
   //  N..(N+C-1): ps threads (N is primary ps thread)
  /** \brief Returns whether the given customer id belongs to a PS thread */
  bool is_ps(const unsigned int customer_id) {
    return num_worker_threads() <= customer_id && customer_id < num_worker_threads() + num_channels();
  }
  /** \brief Returns whether the given customer id belongs to the primary PS thread of this node */
  bool is_primary_ps(const unsigned int customer_id) {
    return num_worker_threads() == customer_id;
  }
  /** \brief Return the customer id for the PS thread of the corresponding channel (`0` gives the primary PS thread) */
  int ps_customer_id(const unsigned int channel) {
    return num_worker_threads() + channel;
  }
  // format the channel number for output
  char fchannel(const unsigned int channel_or_customer_id, const bool given_customer_id=false) const {
    unsigned int channel = channel_or_customer_id;
    if (given_customer_id) {
      assert(channel_or_customer_id >= num_worker_threads());
      channel = channel_or_customer_id - num_worker_threads();
    }
    assert (channel < 52);
    return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[channel];
  }

  // stats and key tracing
  std::string get_stats_output_folder() { return stats_output_folder_; }
  std::string get_traced_keys() { return traced_keys_list_; }


  /**
   * \brief barrier
   * \param node_id the barrier group id
   */
  void Barrier(int customer_id, int node_group);
  /**
   * \brief process a control message, called by van
   * \param the received message
   */
  void Manage(const Message& recv);
  /**
   * \brief update the heartbeat record map
   * \param node_id the \ref Node id
   * \param t the last received heartbeat time
   */
  void UpdateHeartbeat(int node_id, time_t t) {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    heartbeats_[node_id] = t;
  }
  /**
   * \brief get node ids that haven't reported heartbeats for over t seconds
   * \param t timeout in sec
   */
  std::vector<int> GetDeadNodes(int t = 60);


 private:
  Postoffice();
  ~Postoffice() { delete van_; }

  void InitEnvironment();
  Van* van_;
  mutable std::mutex mu_;
  std::vector<Customer*> customers_;
  std::unordered_map<int, std::vector<int>> node_ids_;
  std::mutex server_key_ranges_mu_;
  std::vector<Range> server_key_ranges_;
  bool is_worker_, is_server_, is_scheduler_;
  int num_servers_, num_workers_;
  bool replication_ = false; // [sysChange]
  int num_network_threads_ = 3;
  MgmtTechniques management_techniques_ = MgmtTechniques::ALL;
  bool time_intent_actions_ = true;
  bool location_caches_ = true;
  int num_channels_ = 4;
  std::unordered_map<int, std::unordered_map<int, bool> > barrier_done_;
  int verbose_;
  std::mutex barrier_mu_;
  std::condition_variable barrier_cond_;
  std::mutex heartbeat_mu_;
  std::mutex start_mu_;
  int init_stage_ = 0;
  std::unordered_map<int, time_t> heartbeats_;
  Callback exit_callback_;
  /** \brief Holding a shared_ptr to prevent it from being destructed too early */
  std::shared_ptr<Environment> env_ref_;
  time_t start_time_;
  /*! \brief The maximal allowed key value */
  Key num_keys_ = std::numeric_limits<Key>::max(); // [sysChange] moved here to be able to modify it
  /*! \brief The number of worker threads per server process [sysChange] */
  uint num_worker_threads_ = 1;
  std::string traced_keys_list_;
  std::string stats_output_folder_;
  DISALLOW_COPY_AND_ASSIGN(Postoffice);
};

/** \brief verbose log */
#define PS_VLOG(x) LOG_IF(INFO, x <= Postoffice::Get()->verbose())
}  // namespace ps
#endif  // PS_INTERNAL_POSTOFFICE_H_
