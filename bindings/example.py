#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import numpy as np
from torch.multiprocessing import Process
import lapse
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


def worker_numpy(worker_id, rank, kv):
    """Example worker, using numpy arrays"""
    print("run worker " + str(worker_id) + " on server rank " + str(rank) + ", using NumPy arrays")

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
    kv.localize(keys2)

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



def worker_torch(worker_id, rank, kv):
    """Example worker, using PyTorch tensors """
    print("run worker " + str(worker_id) + " on server rank " + str(rank) + ", using PyTorch tensors")

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
    kv.localize(keys2)

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

def init_scheduler(dummy, num_nodes):
    os.environ['DMLC_NUM_SERVER'] = str(num_nodes)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port

    lapse.scheduler(num_keys, num_workers_per_node)


def init_node(rank, num_nodes):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(num_nodes)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = localip
    os.environ['DMLC_PS_ROOT_PORT'] = port

    lapse.setup(num_keys, num_workers_per_node)

    # in this example, there are `num_keys` keys and all keys except one
    #   hold a vector of length `vpk`. To indicate this to Lapse, we pass
    #   an array of length `num_keys`, in which each key holds the length
    #   of the parameter vector
    value_lengths = torch.ones(num_keys)*vpk
    value_lengths[400] = 10 ## one key holds a vector of other length

    s = lapse.Server(value_lengths)

    threads = []
    for w in range(num_workers_per_node):
        worker_id = rank * num_workers_per_node + w
        kv = lapse.Worker(0, w+1, s)
        t = threading.Thread(target=run_worker, args=(worker_id, rank, kv))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown lapse node
    s.shutdown()


def kill_processes(signal_received, frame):
    """Kills all started lapse processes"""
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)

processes = []
if __name__ == "__main__":
    # catch interrupt (to shut down lapse processes)
    signal(SIGINT, kill_processes)

    # launch lapse scheduler
    p = Process(target=init_scheduler, args=(0, num_nodes))
    p.start()
    processes.append(p)

    # launch lapse processes
    for rank in range(num_nodes):
        p = Process(target=init_node, args=(rank, num_nodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
