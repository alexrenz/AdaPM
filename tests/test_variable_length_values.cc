#include "../apps/utils.h"
#include "ps/ps.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <array>


using namespace ps;


// handle and server types
typedef long ValT;
typedef DefaultColoServerHandle<ValT> HandleT;
typedef ColoKVServer<ValT, HandleT> ServerT;
typedef ColoKVWorker<ValT, HandleT> WorkerT;


// Config
bool coloc = false;
int num_keys = 150;
std::vector<size_t> value_lengths;

vector <int> sim_push_pull_members = {0,2}; //to assign roles of threads in simultaneous_pushpull
bool error = false;

std::string prfx ("Variable-length values: ");


/**
 * function designed to stress relevant keys with Pushes and Pulls. This tests all the Code which forwards/queues these actions during Localize()-Operations, and the eventual correctness
 * The Pushes cancel out and the Pulls are not checked.
 * passing some values not by reference is intentional.
 * should only be entered by  exactly 2 threads, whose id is set in in sim_push_pull-vector
 */
void simultaneous_pushpull(WorkerT &kv, vector<Key> &keys, vector<ValT> values,  int worker_id){
  CHECK(worker_id == sim_push_pull_members[0] || worker_id == sim_push_pull_members[1]) << " wrong worker w["<< worker_id
    << "] has entered this func for it to work properly,\n Allowed workers:: " << sim_push_pull_members;
  int signflip = -1;
  // actions of both of these workers will cancel out this way
  if(worker_id == sim_push_pull_members[0]) signflip = 1;

  // adds  magnitude to clearly distinguish between noise and the real pushes
  for(auto & elem : values) elem = elem * 10 * signflip;

  vector<ValT> pulled_vals(values.size());
  int loop_duration = 9999; // chosen number arbitrarily, hope its large enough to cover the whole duration of the LOCALIZE()-Transfer, and not too long.
  for(int e = 0; e < loop_duration; e++){
      kv.Push(keys, values);
      kv.Pull(keys, &pulled_vals);
  }
  kv.WaitAll();
}

// reverts possible localizes; Each Node localizes its original parameters, this recreates the original parameter distribution.
// Invoked to return to a clean slate in between individual tests.
void reset_kv_localizes(WorkerT &kv){
  int share = num_keys / ps::NumServers();
  vector<Key> keyshare(share);
  std::iota(keyshare.begin(), keyshare.end(), (ps::MyRank() * share));
  kv.Localize(keyshare);
}
/**
    resets/nulls the key-value store in between test functions
    Invoked to return to a clean slate in between these.
*/
void reset_kv_store(WorkerT &kv){
  size_t lenssum = 0;
  //determining the total sum of lengths of all parameters
  for(size_t e = 0; e < value_lengths.size();e ++ ) lenssum += value_lengths[e];
  vector<Key> keys (value_lengths.size());
  vector<ValT> vals (lenssum);
  std::iota(keys.begin(), keys.end(), 0);

  // Pulling all parameters/ the complete state of the key-value store
  kv.Wait(kv.Pull(keys, &vals));
  // inverting these values to update/null the additive store
  for(size_t e = 0; e < vals.size(); e++) {
      vals[e] = -1 * vals[e];
  }
  kv.Wait(kv.Push(keys, vals));
  kv.WaitAll();
  kv.Wait(kv.Pull(keys, &vals));
  for( size_t e = 0; e < vals.size(); e++){
    if(vals[e] != 0) {
      error = true;
      ALOG(prfx << "FAILED. Error during key-value store reset, position " << e << " is not 0:" << vals[e]);
    }
  }
}

/**
 * Single node tests cover parts of the functions Pull/Push, attemptLocal(Push/Pull) and PullIfLocal.
 * No Localize() is tested and no request is transmitted over the network. Purely the single-server functions are checked.
 * Each of these tests use each parameter only once, so that faulty tests can be run individually.
 * The state of each test is always limited to the keys/values/values_pull triplets
 */
void vvpk_single_node(WorkerT &kv, int customer_id){

  std::vector<Key> keys (1);
  std::vector<ValT> values;
  std::vector<ValT> values_pull;

  // single key push
  keys        = {0};
  values      = {1, 2};
  values_pull = {0, 0};

  kv.Wait(kv.Push(keys, values));
  kv.Wait(kv.Pull(keys, &values_pull));
  for(unsigned int e = 0; e< values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Single key Push FAILED. Sent: " << values << ", received: " << values_pull);
    }

  //test two key push; identical parameter length
  keys        = {1, 2};
  values      = {14, 15, 26, 27};
  values_pull = {0, 0, 0, 0};

  kv.Wait(kv.Push(keys, values));
  kv.Wait(kv.Pull(keys, &values_pull));
  for(unsigned int e = 0; e< values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Two key (identical len) Push FAILED. Sent: " << values << ", received: " << values_pull);
    }


  //test two key push: different parameter lengths
  keys        = {3, 34}; // len 2 and 3 respectively
  values      = {31, 32, 341, 342, 343};
  values_pull = {0,   0,   0,   0,   0};

  kv.Wait(kv.Push(keys, values));
  kv.Wait(kv.Pull(keys, &values_pull));
  for(unsigned int e = 0; e< values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Two key (different len) Push FAILED. Sent: " << values << ", received: " << values_pull);
    }

  //test 3 key Push; different parameter lengths
  keys        = {5, 32, 41};
  values      = {51, 52, 321, 322, 323, 411};
  values_pull = { 0,  0,   0,   0,   0,   0};

  kv.Wait(kv.Push(keys, values));
  kv.Wait(kv.Pull(keys, &values_pull));
  for(unsigned int e = 0; e< values.size(); e++)
    if  (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Three key (different len) Push FAILED. Sent: " << values << ", received: " << values_pull);
    }


  //Test PullifLocal
  keys        = {9};
  values      = {99};
  values_pull = {0};
  kv.Wait(kv.Push(keys, values));
  kv.PullIfLocal(keys[0], &values_pull);
  for(unsigned int e = 0; e< values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG("PullIfLocal FAILED. Sent: " << values << ", received: " << values_pull);
    }

  kv.WaitAll();
  if (customer_id == 1 && !error) ALOG(prfx << "All single-node tests PASSED" );
}

/**
 * Dual node tests cover the functions Pull/Push and attemptLocal(Push/Pull) and PullIfLocal,
 * as well as transmitting requests over the network, testing some parts of ProcessPush and ProcessPull.
 * They cover the whole capabilities of traditional key-value-stores.
 * No Localize() is tested.
 * Each of these tests use each parameter only once, so that faulty tests can be run individually.
 * The state of each test is always limited to the keys/values/values_pull triplets
 * Some of these tests are identical to the ones found in vvpk_single_node, but they are invoked from a thread on another server.
 * These are non-local requests.
 *
 * Payloads of each request with different parameter lengths have the following semantic:
 * A payload to parameters x and y, in which x has a value length of 3 and y of 2 is set as following:
 * (x1, x2, x3, y1, y2)
 *
 * Parameter distribution::
 * node 0: 0-49
 * node 1: 50-99
 * node 2: 100-149
 */
void vvpk_dual_node(WorkerT &kv,  int customer_id){
  auto rank = ps::MyRank();

  std::vector<Key> keys(1);
  std::vector<ValT> values;
  std::vector<ValT> values_pull;
  // single key push
  keys        = {0};
  values      = {1, 2};
  values_pull = {0, 0};

  if (rank == 1) kv.Wait(kv.Push(keys, values)); // non-local
  kv.Barrier();

  kv.Wait(kv.Pull(keys, &values_pull));
  for (unsigned int e = 0; e < values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Single key push FAILED on node " << rank << ". Sent: " << values << ", received: " << values_pull);
    }


  kv.Barrier();
  //test two key push; identical length
  keys        = {1, 2};
  values      = {4, 5, 6, 7};
  values_pull = {0, 0, 0, 0};

  if (rank == 1) kv.Wait(kv.Push(keys, values)); // non-local
  kv.Barrier();
  kv.Wait(kv.Pull(keys, &values_pull));
  for (unsigned int e = 0; e < values.size(); e++)
    if (values[e] != values_pull[e]) {
      error = true;
      ALOG(prfx << "Two key (identical len) push FAILED on node " << rank << ". Sent: " << values << ", received: "
      << values_pull);
    }

  kv.Barrier();
  //test two key push: different parameter lengths which are not co-located (-> request is split and sent out to two nodes)
  keys        = {42, 61};
  values      = {421, 422, 611};
  values_pull = {  0,   0,   0};

  kv.Wait(kv.Push(keys, values));
  kv.WaitAll();
  kv.Barrier();
  kv.Wait(kv.Pull(keys, &values_pull));
  kv.WaitAll();
  kv.Barrier();
  for (unsigned int e = 0; e < values.size(); e++)
    if (values[e] * 3 != values_pull[e]) {
      error = true;
      ALOG(prfx << "Two key (different len, mixed locality) push FAILED on node " << rank << ". Sent: " << values
      << ", received: " << values_pull);
    }

  kv.Barrier();
  //test 3 keys with different lengths, but same location
  keys        = {50, 80, 72};
  values      = {501, 502, 503, 801, 721, 722};
  values_pull = {  0,   0,   0,   0,   0,   0};

  if (rank == 0) kv.Wait(kv.Push(keys, values));
  kv.Barrier();
  kv.Wait(kv.Pull(keys, &values_pull));
  for (unsigned int e = 0; e < values.size(); e++)
     if (values[e] != values_pull[e]) {
       error = true;
       ALOG(prfx << "Three key (different len, same non-locality) push FAILED on node " << rank << ". Sent: " << values
       << ", received: " << values_pull);
     }

  kv.Barrier();
  //Test PullifLocal
  keys        = {90};
  values      = {99};
  values_pull = {0};

  if (rank == 1) kv.Wait(kv.Push(keys, values));
  if (kv.PullIfLocal(keys[0], &values_pull)) {
    if (rank != 1) {
      error = true;
      ALOG(prfx << "FAILED. Node " << rank << " should not have key " << keys[0]);
    }
    for (unsigned int e = 0; e < values.size(); e++)
      if (values[e] != values_pull[e]) {
        error = true;
        ALOG(prfx << "PullIfLocal FAILED on node " << rank << ". Sent: " << values << ", received: " << values_pull);
      }
  }
  kv.Barrier();
  if (rank == 0 && !error) ALOG(prfx << "All dual node tests (trad. PS store) PASSED" );
}

  /**
  * This  tests the Localize() functionality.
  * This includes the locality as well as the correctness of transmission.
  * The test also commits dummy- Pull() and Push() requests during the Localize(), to test eventual consistency,
  * as well as tolerance to concurrent execution.
  * Each of these tests use each parameter only once, so that faulty tests can be run individually.
  * The state of each test is always limited to the keys/values/values_pull triplets
  *
  * Payloads of each request with different parameter lengths have the following semantic:
  * A payload to parameters x and y, in which x has a value length of 3 and y of 2 is set as following:
  * (x1, x2, x3, y1, y2)
  *
  * Parameter distribution::
  * node 0: 0-49
  * node 1: 50-99
  * node 2: 100-149
  */
void vvpk_localize_contention(WorkerT &kv, int customer_id){
  sim_push_pull_members = {0,2};
  int worker_id = ps::MyRank() + customer_id - 1; // a unique id for this worker thread
  std::vector<Key> keys (1);
  std::vector<ValT> values;
  std::vector<ValT> values_pull;

  keys   = {0};
  values = {10, 11};
  //TEST 1:: single key,
  kv.Push(keys, values);
  if (worker_id == 1) kv.Localize(keys);

  kv.WaitAll();
  kv.Barrier();

  values      = {30, 33}; // 3 threads pushed {10, 11}
  values_pull = {0,   0};

  //checking locality of parameter
  for (unsigned int e = 0; e < keys.size(); e++) {
    Key key = keys[e];
    if (worker_id == 1) {
      if (!kv.PullIfLocal(key, &values_pull)) {
        error = true;
        ALOG(prfx << "FAILED. Key " << keys << " has not migrated");
      }
    } else {
      kv.Wait(kv.Pull(vector <Key> {key}, &values_pull));
    }
  }

  kv.WaitAll();
  kv.Barrier();
  //checking values of parameter
  for (unsigned int e = 0; e < values_pull.size(); e++)
    if (values_pull[e] != values[e]) {
      error = true;
      ALOG(prfx << "FAILED. Pulled " << values_pull << ", but should be " << values);
    }

  kv.Barrier();
  // Test of a single Localize() with concurrent Pushes and Pulls.
  keys = {140};
  values      = {140, 141, 142};
  values_pull = {  0,   0,   0};

  if(worker_id == 2) kv.Wait(kv.Push(keys, values));

  kv.Barrier();

  if(worker_id == 1) kv.Localize(keys); //Localize() parameter from node 2 to 1, with node 0 & 2 spamming pushes and pulls
  else               simultaneous_pushpull(kv, keys, values, worker_id);


  kv.WaitAll();
  kv.Barrier();
  //checking location and values of parameter
  if (worker_id == 1) {
    if (!kv.PullIfLocal(keys[0], &values_pull)) {
      error = true;
      ALOG(prfx << "FAILED. Parameter " << keys << " did not migrate");
    }
    for (unsigned int j = 0; j < values_pull.size(); j++)
      if (values_pull[j] != values[j]) {
        error = true;
        ALOG(prfx << "FAILED. Input " << values
            << " differs from pulled " << values_pull
            << " at position " << j);
    }
}


  kv.Barrier();
  //Tests 3  of Localizes with multiple Parameters, each with different len
  //Localize() from node 0 to node 1, node 0 and 2 spam pushes and pulls
  keys = {10, 30, 20, 40};
  values      = {11, 12, 31, 32, 33, 21, 22, 41, 42, 43};
  values_pull = { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

  if (worker_id == 1) kv.Wait(kv.Push(keys, values));

  kv.Barrier();
  if (worker_id == 1) kv.Localize(keys);
  else                simultaneous_pushpull(kv, keys, values, worker_id);

  kv.WaitAll();
  kv.Barrier();

  //checking location and values of keys
  if (worker_id == 1) {
    int lens_offset = 0;
    for (unsigned int e = 0; e < keys.size(); e++) {
      Key key = keys[e];
      values_pull.clear();
      values_pull.resize(value_lengths[key], 0);
      if (!kv.PullIfLocal(key, &values_pull)) {
        error = true;
        ALOG(prfx << "FAILED. Parameter " << keys << " has not migrated in between nodes");
      }
      for (unsigned int j = 0; j < values_pull.size(); j++)
        if (values_pull[j] != values[lens_offset + j]) {
          error = true;
          ALOG(prfx << "FAILED. Input " << values
               << " differs from pulled " << values_pull
               << " at position Key e=" << e << " valposition:" << lens_offset + j
               << " expected :" << values[lens_offset+j]);
        }
      lens_offset += value_lengths[key];
    }
  }

  kv.WaitAll();
  kv.Barrier();

  //Test of localizes with multiple parameters with different destinations
  // node 0 requesting keys from node 1, 2
  keys = {100, 50, 110, 60};
  values      = {10, 11, 12, 51, 52, 53, 110, 111, 112, 61, 62, 63};
  values_pull = {0,   0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0};
  if(worker_id == 0) kv.Wait(kv.Push(keys, values));

  kv.Barrier();

  sim_push_pull_members = {1,2};// node 0 Localizes, node 1 and 2 spam, this vector enforces that.
  if (worker_id == 0) kv.Localize(keys);
  else                simultaneous_pushpull(kv, keys, values, worker_id);

  kv.WaitAll();
  kv.Barrier();

  if (worker_id == 0) {
    int lens_offset = 0;
    for (unsigned int e = 0; e < keys.size(); e++) {
      Key key = keys[e];
      values_pull.clear();
      values_pull.resize(value_lengths[key], 0);
      if (!kv.PullIfLocal(key, &values_pull)) {
        error = true;
        ALOG(prfx << "FAILED. Parameter " << keys << " has not migrated in between nodes");
      }
      for (unsigned int j = 0; j < values_pull.size(); j++)
        if (values_pull[j] != values[lens_offset + j]) {
          error = true;
          ALOG(prfx << "Pull of multiple non-uniform localized parameters FAILED. Input "
               << values << " differs from returned output " << values_pull);
        }
      lens_offset += value_lengths[key];
    }
  }

  // TEST in which parameter is passed from node 1 to 0 to 2
  // successive Localize()
  kv.Barrier();
  keys        = {70};
  values      = {777};
  values_pull = {0};

  if(worker_id == 0){ //setup
    kv.Wait(kv.Push(keys,values));
    kv.Localize(keys);
    kv.WaitAll();
    if (!kv.PullIfLocal(keys[0], &values_pull)) {
      error = true;
      ALOG(prfx << "Initial Localize in successive Localize()-test FAILED "); // param needs to be local now
    }
  }
  kv.Barrier();
  if(worker_id == 2 && (kv.PullIfLocal(keys[0], &values_pull))) {
    error = true;
    ALOG(prfx << "Initial Localize in successive Localize()-test FAILED "); // param should not be local on this node
  }
  kv.Barrier();
  if(worker_id == 2) kv.Localize(keys);
  kv.WaitAll();
  kv.Barrier();
  if(worker_id == 2) {
    if (!kv.PullIfLocal(keys[0], &values_pull)) {
      error = true;
      ALOG(prfx << " Second Localize in successive Localize()-test FAILED "); // param needs to be local on this node now
    }
  }  else {
    if ((kv.PullIfLocal(keys[0], &values_pull))) {
      error = true;
      ALOG(prfx << " second Localize in successive Localize()-test FAILED. Key is apparently still local on node "
      << worker_id); // param should not be local
    }
  }
  kv.Barrier();
  if (worker_id == 0 && !error) ALOG(prfx << "All Localize tests with concurrent Pushes/Pulls PASSED" );
}

  /**
  * The tests below test the Localize() functionality.
  * This includes the locality as well as the correctness of transmission.
  *
  * Each of these tests use each parameter only once, so that faulty tests can be run individually.
  * The state of each test is always limited to the keys/values/values_pull triplets
  *
  * Payloads of each request with different parameter lengths have the following semantic:
  * A payload to parameters x and y, in which x has a value length of 3 and y of 2 is set as following:
  * (x1, x2, x3, y1, y2)
  *
  * Parameter distribution::
  * node 0: 0-49
  * node 1: 50-99
  * node 2: 100-149
  */
void vvpk_localize_nc(WorkerT &kv, int customer_id){
    sim_push_pull_members = {0,2};
    auto rank = ps::MyRank();

    int worker_id = ps::MyRank() + customer_id - 1; // a unique id for this worker thread
    std::vector<Key> keys (1);
    std::vector<ValT> values;
    std::vector<ValT> values_pull;

    keys = {0};
    values = {11, 12};
    // TEST 1:: single key, no lens
    kv.Wait(kv.Push(keys, values));

    kv.Barrier();

    if (rank == 1) kv.Localize(keys);

    kv.WaitAll();
    kv.Barrier();

    values_pull = {0, 0};
    values        = {3 * 11, 3 * 12}; // 3 threads Push() in this test

    //check of locality and values
    if (rank == 1) {
      for (size_t e = 0; e < keys.size(); e++) {
        Key key = keys[e];
        if (!kv.PullIfLocal(key, &values_pull)) {
          error = true;
          ALOG(prfx << "Localize has FAILED , parameter " << keys
          << " has not migrated in between nodes");
        }
      }
      for (size_t e = 0; e < values_pull.size(); e++)
        if (values_pull[e] != values[e]) {
          error = true;
          ALOG(prfx << "Pull of single localized parameter FAILED ; input " << values
          << " differs from returned output " << values_pull);
        }
    }
    kv.Barrier();

    //Test 2  of Localizes with multiple Parameters, with different lengths
    keys        = {10, 20, 30, 40};
    values      = {11, 12, 21, 22, 31, 32, 33, 41, 42, 43};
    values_pull = { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

    if(rank == 1) kv.Wait(kv.Push(keys, values));
    kv.WaitAll();
    kv.Barrier();
    if(rank == 1) kv.Localize(keys);
    kv.WaitAll();
    kv.Barrier();
    int lens_offset = 0;
    //checking locality and values
    // each key is checked individually via PullIfLocal
    for (size_t e = 0; e < keys.size(); e++) {
      Key key = keys[e];
      values_pull.clear();
      values_pull.resize(value_lengths[keys[e]], 0);

      if (rank == 1){
         if (!kv.PullIfLocal(key, &values_pull)) {
           error = true;
           ALOG(prfx << " Localize has FAILED , parameter " << keys
           << " has not migrated in between nodes");
         }
      } else {
        kv.Wait(kv.Pull(vector <Key> {key}, &values_pull));
      }

      for (size_t j = 0; j < values_pull.size(); j++)
        if (values_pull[j] != values[lens_offset + j]) {
          error = true;
          ALOG(prfx << " Pull of multiple localized parameters FAILED on node " << rank <<"; key " << key << " input "
                  << values << " differs from pulled " << values_pull << ". Affected pos: "
                  << values[lens_offset + j] << " differs from returned " << values_pull[j]
                  << " with keys " << vector<Key> {key});
        }
      lens_offset += value_lengths[keys[e]];
    }


    kv.Barrier();
    // Test of uniform subset of these pushed keys.
    keys        = {30, 40};
    values      = {31, 32, 33, 41, 42, 43};
    values_pull = { 0,  0,  0,  0,  0,  0};
    lens_offset = 0;
    for (size_t e = 0; e < keys.size(); e++) {
      Key key = keys[e];
      values_pull.clear();
      values_pull.resize(value_lengths[keys[e]], 0);

      if (rank == 1) {
        if (!kv.PullIfLocal(key, &values_pull)) {
          error = true;
          ALOG(prfx << "Localize has FAILED , parameter " << keys
          << " has not migrated in between nodes");
        }
      } else {
        kv.Wait(kv.Pull(vector < Key > {key}, &values_pull));
      }

      for (size_t j = 0; j < values_pull.size(); j++)
        if (values_pull[j] != values[j + lens_offset]) {
          error = true;
          ALOG(prfx << " Pull of multiple localized parameters FAILED on node " << rank << "; input " << values
          << " differs from returned output " << values_pull);
        }
      lens_offset += value_lengths[keys[e]];
    }

    kv.Barrier();

    //Test of localizes with multiple parameters with different destinations
    keys = {80, 90, 100, 110}; // the keys lie in node 0 and node 2 respectively; 1-1-3-3 lengths
    values      = {81, 91, 101, 102, 103, 111, 112, 113};
    values_pull = { 0,  0,   0,   0,   0,   0,   0,   0};
    kv.Wait(kv.Push(keys, values));
    kv.WaitAll();
    kv.Barrier();
    if(rank == 0) kv.Localize(keys);
    kv.WaitAll();
    kv.Barrier();


    lens_offset = 0;
    //checking locality and values individually
    for (size_t e = 0; e < keys.size(); e++) {
      Key key = keys[e];
      values_pull.clear();
      values_pull.resize(value_lengths[key], 0);
      if(rank == 0) {
        if (!kv.PullIfLocal(key, &values_pull)) {
          error = true;
          ALOG(prfx << "Localize has FAILED , parameter " << keys[e]
          << " has not migrated in between nodes");
        }
      } else {
        kv.Wait(kv.Pull(vector < Key > {key}, &values_pull));
      }

      for (size_t j = 0; j < values_pull.size(); j++)
        if (values_pull[j] != 3 * values[lens_offset + j]) {
          error = true;
          ALOG(prfx << " Pull of multiple non-uniform localized parameters FAILED; input "
          << values << " (x3) differs from returned output " << values_pull);
        }
      lens_offset += value_lengths[key];
    }


    kv.WaitAll();
    kv.Barrier();
    if (worker_id == 0 && !error) ALOG(prfx << "All localize tests in isolation PASSED" );
}

template <typename Val>
void RunWorker(int customer_id, bool barrier, ServerT* server=nullptr) {
  Start(customer_id);
  WorkerT kv(0, customer_id, *server); // app_id, customer_id

  // wait for all workers to boot up
  kv.Barrier();

  auto rank = ps::MyRank();
  int worker_id = ps::MyRank() + customer_id - 1; // a unique id for this worker thread

  //first tests; single server only
  if(rank == 0) vvpk_single_node(kv, customer_id);

  kv.WaitAll();
  kv.Barrier();
  if(worker_id == 0) reset_kv_store(kv);
  kv.WaitAll();
  kv.Barrier();
  // second suite of tests; traditional key-value server capabilities
  vvpk_dual_node(kv, customer_id);

  kv.WaitAll();
  kv.Barrier();
  if(worker_id == 0) reset_kv_store(kv);
  kv.WaitAll();
  kv.Barrier();

  //third: testing Localize() in isolation
  vvpk_localize_nc(kv, customer_id);
  kv.WaitAll();
  kv.Barrier();
  if(worker_id == 0) reset_kv_store(kv);
  kv.WaitAll();
  kv.Barrier();
  if(customer_id == 1) reset_kv_localizes(kv); //done on each node individually
  kv.WaitAll();
  kv.Barrier();

  //fourth; testing Localize() with concurrent Pushes and Pulls.
  // Testing eventual consistency and concurrent execution.
  vvpk_localize_contention(kv, customer_id);

  kv.Finalize();
  Finalize(customer_id, barrier);
  ADLOG("Worker " << rank << ":" << customer_id << " passed finalize ");
}

int main(int argc, char *argv[]) {
  int num_local_workers = 1;
  num_keys = 150;
  value_lengths.resize(num_keys);
  std::fill(value_lengths.begin(), value_lengths.begin() + 30, 2); // first 30(0x-1x-2x) parameters are len 2
  std::fill(value_lengths.begin() + 30 , value_lengths.begin() + 60, 3); // middle 30 (3x-4x-5x) are len 3
  std::fill(value_lengths.begin() + 60, value_lengths.begin() + 100, 1); // next 40 (6x-7x-8x-9x) are len 1
  std::fill(value_lengths.begin() + 100, value_lengths.end(), 3); // last 40 (10x-11x-12x-13x-14x) are len 3
  value_lengths[41] = 1; // extra settings for single_node tests
  value_lengths[72] = 2; //extra setting dual node
  Postoffice::Get()->enable_dynamic_allocation(num_keys, num_local_workers, false);

  // Colocate servers and workers into one process?
  coloc = true;
  std::string role = std::string(getenv("DMLC_ROLE"));

  // co-locate server and worker threads into one process
  if (role.compare("scheduler") == 0) {
    Start(0);
    Finalize(0, true);
  } else if (role.compare("server") == 0) { // worker+server

    // Start the server system
    int server_customer_id = 0; // server gets customer_id=0, workers 1..n
    Start(server_customer_id);
    HandleT handle (value_lengths.size(), value_lengths); // the handle specifies how the server handles incoming Push() and Pull() calls
    auto server = new ServerT(server_customer_id, handle);
    RegisterExitCallback([server](){ delete server; });

    // run worker(s)
    std::vector<std::thread> workers {};
    for (int i=0; i!=num_local_workers; ++i)
      workers.push_back(std::thread(RunWorker<ValT>, i+1, false, server));

    // wait for the workers to finish
    for (size_t w=0; w!=workers.size(); ++w) {
      workers[w].join();
      ADLOG("Customer r" << Postoffice::Get()->my_rank() << ":c" << w+1 << " joined");
    }

    // stop the server
    server->shutdown();
    Finalize(server_customer_id, true);

  } else {
    LL << "Process started with unknown role '" << role << "'.";
  }


  CHECK(!error) << "Test FAILED ";

  return error;
}
