/*!
 *  Copyright (c) 2015 by Contributors
 * @file   ps.h
 * \brief  The parameter server interface
 */
#ifndef PS_PS_H_
#define PS_PS_H_
/** \brief basic setups in ps */
#include "ps/base.h"
/** \brief communicating with a pair of (int, string). */
#include "ps/simple_app.h"
/** \brief communcating with a list of key-value pairs. */
#include "ps/kv_app.h"
/** \brief communcating with a list of key-value pairs. servers co-located with workers. */
#include "ps/coloc_kv_worker.h"
#include "ps/coloc_kv_server.h"
#include "ps/coloc_kv_server_handle.h"
namespace ps {
/** \brief Returns the number of worker nodes */
inline int NumWorkers() { return Postoffice::Get()->num_workers(); }
/** \brief Returns the number of server nodes */
inline int NumServers() { return Postoffice::Get()->num_servers(); }
/** \brief Returns true if this node is a worker node */
inline bool IsWorker() { return Postoffice::Get()->is_worker(); }
/** \brief Returns true if this node is a server node. */
inline bool IsServer() { return Postoffice::Get()->is_server(); }
/** \brief Returns true if this node is a scheduler node. */
inline bool IsScheduler() { return Postoffice::Get()->is_scheduler(); }
/** \brief Returns the rank of this node in its group
 *
 * Each worker will have a unique rank within [0, NumWorkers()). So are
 * servers. This function is available only after \ref Start has been called.
 */
inline int MyRank() { return Postoffice::Get()->my_rank(); }
/**
 * \brief Run the scheduler.
 */
inline void Scheduler() {
  auto scheduler_customer_id = Postoffice::Get()->ps_customer_id(0); // scheduler has the same customer id as the primary PS threads
  Postoffice::Get()->Start(scheduler_customer_id, nullptr, true);
  Postoffice::Get()->Finalize(scheduler_customer_id, true);
}
/**
  * \brief Setup the PS system.
  */
inline void Setup(const Key num_keys, const unsigned int num_threads) {
  Postoffice::Get()->setup(num_keys, num_threads);
}
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
inline void RegisterExitCallback(const std::function<void()>& cb) {
  Postoffice::Get()->RegisterExitCallback(cb);
}

}  // namespace ps
#endif  // PS_PS_H_
