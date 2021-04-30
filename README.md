![lapse logo](docs/lapse2.svg?raw=true) 

[![Build Status](https://travis-ci.org/alexrenz/lapse-ps.svg?branch=main)](https://travis-ci.org/alexrenz/lapse-ps/)
![Build on Latest Ubuntu](https://github.com/alexrenz/lapse-ps/workflows/Latest%20Ubuntu/badge.svg)
[![GitHub license](docs/apache2.svg?raw=true)](./LICENSE)

Lapse2 is a parameter server that combines multiple techniques to manage parameters. It replicates a small set of hot spot parameters, and relocates all other parameters. Combining multiple management techniques is key for efficient distributed training of many ML tasks. More information can be found in our paper on multi-technique parameter servers and Lapse2 ([arXiv.org](https://arxiv.org/abs/2104.00501)).
Details on the **experiment settings** for this paper can be found in [docs/experiments-arxiv21.md](docs/experiments-arxiv21.md).

The implementation of Lapse2 implementation is based on [Lapse1](https://github.com/alexrenz/lapse-ps/releases/tag/v1.0) and [PS-Lite](https://github.com/dmlc/ps-lite).

Lapse2 provides the following primitives: 
- `Pull(keys)`: retrieve the values of a set of parameters (identified by keys) from the corresponding servers 
- `Push(keys, updates)`: send updates for parameters to the corresponding servers
- `Localize(keys)`: request local allocation of parameters
- `handle = PrepareSample(N)`: prepare a group of `N` samples
- `PullSample(handle)`: retrieve `n` samples from a prepared group

By default, primitives execute asynchronously. `Wait()` can be used to execute any primitive (except `PrepareSample`) synchronously. For example: `Wait(Pull(keys))`.


A simple example:

```c++
  std::vector<uint64_t> keys = {1, 3, 5};
  std::vector<float> updates = {1, 1, 1};
  std::vector<float> vals;
  ps::KVWorker<float> kv;

  kv.Wait(kv.Pull(keys, &vals));
  kv.Wait(kv.Push(keys, updates));

  kv.Wait(kv.Localize(keys));
  kv.Wait(kv.Pull(keys, &vals)); // access is now local
  
  auto h = kv.PrepareSample(3); // prepare a group of 20 samples
  kv.Wait(kv.PullSample(h, keys, vals)); // pull the 3 samples (keys.size() determines how many samples are pulled)
```

### Build

`lapse2` requires a C++11 compiler such as `g++ >= 4.8` and boost for some the application examples. On Ubuntu >= 13.10, you
can install it by
```
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
```

Then clone and build

```bash
git clone https://github.com/alexrenz/lapse-ps
cd lapse-ps && make
```

### Getting started

A very simple example can be found in [simple.cc](apps/simple.cc). To run it, compile it:

```bash
make apps/simple
```

and run

```bash
python tracker/dmlc_local.py -s 1 build/apps/simple
```

to run with one node and default parameters or 

```bash
python tracker/dmlc_local.py -s 3 build/apps/simple -v 5 -i 10 -k 14 -t 4
```
to run with 3 nodes and specific parameters. Run `build/apps/simple --help` to see available parameters.

### Testing
To test dynamic parameter allocation (i.e., moving parameters between servers), run

```bash
make -j 4 tests/test_dynamic_allocation
python tracker/dmlc_local.py -s 4 tests/test_dynamic_allocation
```



### Starting an application

There are multiple start scripts. At the moment, we mostly use the following ones:
- [tracker/dmlc_local.py](tracker/dmlc_local.py) to run on a local machine
- [tracker/dmlc_ssh.py](tracker/dmlc_ssh.py) to run on a cluster
To see more information, run `python tracker/dmlc_local.py --help`, for example.

The `-s` flag specifies how many processes (i.e., nodes to use, e.g. `-s 4` uses 4 nodes. In each process, Lapse2 starts one server thread and multiple worker threads. 

### Example Applications

You find example applications in the [apps/](apps/) directory and launch commands to locally run toy examples below. The toy datasets are in [apps/data/](apps/data/). 


#### Knowledge Graph Embeddings
```
make apps/knowledge_graph_embeddings
python tracker/dmlc_local.py -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 100 --eval_freq 2
```

#### Word vectors
```
make apps/word2vec
python tracker/dmlc_local.py -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --localize_pos 1 --data_words 10000
```

#### Matrix Factorization

```
make apps/matrix_factorization
python tracker/dmlc_local.py -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 2 --num_rows 6 --num_cols 4 --epochs 10
```

### Architecture

Lapse2 starts one process per node. Within this process, worker threads access the parameter store directly. A parameter server thread handles requests by other nodes and parameter relocations.

![architecture](docs/architecture.png?raw=true)

