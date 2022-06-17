/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_BASE_H_
#define PS_BASE_H_
#include <limits>
#include "ps/internal/utils.h"
namespace ps {

#ifdef PS_KEY_TYPE
/*! \brief Use the specified key type */
using Key = PS_KEY_TYPE;
#else
/*! \brief Default: use unsigned 64-bit int as the key type */
using Key = uint64_t;
#endif

using Clock = long;
static const Clock CLOCK_MAX = std::numeric_limits<Clock>::max()-1;
static const Clock WORKER_FINISHED = std::numeric_limits<Clock>::max();
static const Clock WINDOW_MAX = CLOCK_MAX/2;

using Version = int64_t; // for replicas, see handle

/** \brief node ID for the scheduler */
static const int kScheduler = 1;
/**
 * \brief the server node group ID
 *
 * group id can be combined:
 * - kServerGroup + kScheduler means all server nodes and the scheuduler
 * - kServerGroup + kWorkerGroup means all server and worker nodes
 */
static const int kServerGroup = 2;
/** \brief the worker node group ID */
static const int kWorkerGroup = 4;
/** \brief group for all worker threads */
static const int kWorkerThreadGroup = 0;
/* warning! other than kServerGroup and kWorkerGroup, kWorkerThreadGroup
   is a group of threads (and not processes). Adding to other groups does
   work and it works only for the co-located case.
*/

/* Type of an operation */
enum class OpType {PULL=0, PUSH=1, SET=2};

/* ID/handle for a sample */
using SampleID = size_t;

/* Indicates that a parameter is locally available at the current node */
const int LOCAL = -1;

/* Options for management technique choices */
enum class MgmtTechniques {ALL, REPLICATION_ONLY, RELOCATION_ONLY};

/* Pair of a customer id and a clock (e.g., for passing around intent ends) */
struct WorkerClockPair {
  WorkerClockPair(const int customer_id_, const Clock e):
    customer_id{customer_id_}, end{e} {}
  int customer_id;
  Clock end;
};

}  // namespace ps
#endif  // PS_BASE_H_
