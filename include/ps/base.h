/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_BASE_H_
#define PS_BASE_H_
#include "ps/internal/utils.h"
namespace ps {

#if USE_KEY32
/*! \brief Use unsigned 32-bit int as the key type */
using Key = uint32_t;
#else
/*! \brief Use unsigned 64-bit int as the key type */
using Key = uint64_t;
#endif
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

/* Status of a parameter: it resides in a local PS, is in transfer to the local PS, or resides on a remote PS */
 enum Status {LOCAL, IN_TRANSFER, REMOTE}; // [sysChange]: status enum

/* ID/handle for a sample */
using SampleID = size_t;

}  // namespace ps
#endif  // PS_BASE_H_
