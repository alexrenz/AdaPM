#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import numpy as np
from torch.multiprocessing import Process
import adaps
from signal import signal, SIGINT
from sys import exit
import threading

num_nodes = 4    # number of nodes
num_workers_per_node = 2 # number of worker threads per node
num_keys = 1000  # number of keys
vpk = 2          # length of the parameter vector that one key holds

localip = '127.0.0.1'
port = '9091'

def run_worker(worker_id, rank, kv):
    if worker_id == 0:
        print("""\n
---------------------------------------------------
Run example with PyTorch tensors
----------------------------------------------------
""")

    worker_torch(worker_id, rank, kv)
    kv.barrier() # wait for all workers to finish



    if worker_id == 0:
        print("""\n
---------------------------------------------------
Run example with NumPy arrays
----------------------------------------------------
""")
    worker_numpy(worker_id, rank, kv)
    kv.barrier() # wait for all workers to finish

    kv.finalize()


def worker_numpy(worker_id, rank, kv):
    """Example worker, using numpy arrays"""
    print("run worker " + str(worker_id) + " on server rank " + str(rank) + ", using NumPy arrays")

    try:
        np.random.seed(worker_id)

        keys = np.array([1,2,3,4])
        keys2 = np.array([1,333,666,960])+worker_id
        vals = np.ones((len(keys)*vpk), dtype=np.float32)
        pushvals = np.random.rand(len(keys2)*vpk).astype(np.float32)
        setvals = np.ones((len(keys)*vpk), dtype=np.float32)

        # pull
        kv.pull(keys, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals))

        # localize
        kv.intent(keys2, kv.current_clock()+1)

        kv.advance_clock()
        kv.wait_sync()

        # push
        print("worker " + str(worker_id) + " pushes " + str(pushvals))
        kv.push(keys2, pushvals)

        # pull to check values
        kv.pull(keys2, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals) + " after push")

        # set
        kv.set(keys2, setvals)

        # asynchronous operations
        kv.push(keys2, pushvals, True)
        kv.pull(keys2, pushvals, True)

        # pull to check values
        kv.pull(keys2, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals) + " after set")

        # asynchronous operations
        ts1 = kv.push(keys2, pushvals, True)
        ts2 = kv.pull(keys2, vals, True)
        kv.wait(ts1) # optional
        kv.wait(ts2) # optional

        ## pull the key that holds a vector of other length
        longer_key = np.array([400])
        longer_vals = np.ones((10), dtype=np.float32)
        kv.pull(longer_key, longer_vals)

        ## sampling
        N = 8
        s1 = kv.prepare_sample(N, kv.current_clock())
        samplekeys = np.zeros(N, dtype=np.int64)
        samplevals = np.ones((len(samplekeys)*vpk), dtype=np.float32)
        kv.pull_sample(s1, samplekeys, samplevals)
        print("sampled keys in w" + str(worker_id) + ": " + str(samplekeys))
    except Exception as e:
        print(e)
        os._exit(1)



def worker_torch(worker_id, rank, kv):
    """Example worker, using PyTorch tensors """
    print("run worker " + str(worker_id) + " on server rank " + str(rank) + ", using PyTorch tensors")

    try:
        np.random.seed(worker_id)
        torch.manual_seed(worker_id)

        keys = torch.LongTensor([1,2,3,4])
        keys2 = torch.LongTensor([1,333,666,960])+worker_id
        vals = torch.ones((len(keys)*vpk), dtype=torch.float32)
        pushvals = torch.from_numpy(np.random.rand(len(keys2)*vpk).astype(np.float32))
        setvals = torch.ones((len(keys)*vpk), dtype=torch.float32)

        # pull
        kv.pull(keys, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals))

        # localize
        kv.intent(keys2, kv.current_clock()+1)

        kv.advance_clock()
        kv.wait_sync()

        # push
        print("worker " + str(worker_id) + " pushes " + str(pushvals))
        kv.push(keys2, pushvals)

        # pull to check values
        kv.pull(keys2, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals) + " after push")

        # set
        kv.set(keys2, setvals)

        # pull to check values
        kv.pull(keys2, vals)
        print("worker " + str(worker_id) + " pulled " + str(vals) + " after set")

        # asynchronous operations
        ts1 = kv.push(keys2, pushvals, True)
        ts2 = kv.pull(keys2, vals, True)
        kv.wait(ts1) # optional
        kv.wait(ts2) # optional

        ## pull the key that holds a vector of other length
        longer_key = torch.LongTensor([400])
        longer_vals = torch.ones((10), dtype=torch.float32)
        kv.pull(longer_key, longer_vals)

    except Exception as e:
        print(e)
        os._exit(1)

def init_scheduler(dummy, num_nodes):
    os.environ['DMLC_NUM_SERVER'] = str(num_nodes)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port

    adaps.scheduler(num_keys, num_workers_per_node)


def init_node(rank, num_nodes):
    """Start up an AdaPS node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(num_nodes)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port

    adaps.setup(num_keys, num_workers_per_node)

    # in this example, there are `num_keys` keys and all keys except one
    #   hold a vector of length `vpk`. To indicate this to AdaPS, we pass
    #   an array of length `num_keys`, in which each key holds the length
    #   of the parameter vector
    value_lengths = torch.ones(num_keys)*vpk
    value_lengths[400] = 10 ## one key holds a vector of other length

    s = adaps.Server(value_lengths)
    s.enable_sampling_support(scheme="local", with_replacement=True,
                              distribution="uniform", min=0, max=int(num_keys/2))

    threads = []
    for w in range(num_workers_per_node):
        worker_id = rank * num_workers_per_node + w
        t = threading.Thread(target=run_worker, args=(worker_id, rank, adaps.Worker(w, s)))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown AdaPS node
    s.shutdown()


def kill_processes(signal_received, frame):
    """Kills all started AdaPS processes"""
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)

processes = []
if __name__ == "__main__":
    # catch interrupt (to shut down AdaPS processes)
    signal(SIGINT, kill_processes)

    # launch AdaPS scheduler
    p = Process(target=init_scheduler, args=(0, num_nodes))
    p.start()
    processes.append(p)

    # launch AdaPS processes
    for rank in range(num_nodes):
        p = Process(target=init_node, args=(rank, num_nodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
