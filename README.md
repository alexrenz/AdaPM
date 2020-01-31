![lapse logo](docs/lapse.png?raw=true) 

[![Build Status](https://travis-ci.com/alexrenz/lapse.svg?token=qPF2yxPz6mVQ9DGSToqy&branch=master)](https://travis-ci.com/alexrenz/lapse)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

A parameter server with dynamic parameter allocation. I.e., it can relocate parameters among nodes during run time. This enables efficient distributed machine learning. Lapse provides the following primitives: 
- `Pull(keys)`: retrieve the values of a set of parameters (identified by keys) from the corresponding servers 
- `Push(keys, updates)`: send updates for parameters to the corresponding servers
- `Localize(keys)`: request local allocation of parameters

By default, primitives execute asynchronously. `Wait()` can be used to execute any primitive synchronously. For example: `Wait(Pull(keys))`.

The Lapse implementation is based on [PS-Lite](https://github.com/dmlc/ps-lite).

A simple example:

```c++
  std::vector<uint64_t> keys = {1, 3, 5};
  std::vector<float> updates = {1, 1, 1};
  std::vector<float> recv_vals;
  ps::KVWorker<float> kv;

  kv.Wait(kv.Pull(keys, &recv_vals));
  kv.Wait(kv.Push(keys, updates));

  kv.Wait(kv.Localize(keys));
  kv.Wait(kv.Pull(keys, &recv_vals)); // access is now local
```

### Build

`lapse` requires a C++11 compiler such as `g++ >= 4.8` and boost for some the application examples. On Ubuntu >= 13.10, you
can install it by
```
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
```

Then clone and build

```bash
git clone https://github.com/alexrenz/lapse
cd lapse && make
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

The `-s` flag specifies how many processes (i.e., nodes to use, e.g. `-s 4` uses 4 nodes. In each process, Lapse starts one server thread and multiple worker threads. 

### Example Applications

You find example applications in the [apps/](apps/) directory and launch commands to locally run toy examples below. The toy datasets are in [apps/data/](apps/data/). 


#### Matrix Factorization

```
make apps/dsgd
python tracker/dmlc_local.py -s 2  build/apps/dsgd --dataset apps/data/mf/ -r 2 --num_keys 12 --epochs 10
```

#### Knowledge Graph Embeddings
```
make apps/knowledge_graph_embeddings
python tracker/dmlc_local.py -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 100 --eval_freq 2
```

#### Language Modeling
```
make apps/word2vec
python tracker/dmlc_local.py -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --localize_pos 1 --localize_neg 1 --data_words 10000
```

### Architecture

Lapse starts one process per node. Within this process, worker threads access the parameter store directly. A parameter server thread handles requests by other nodes and parameter relocations.

![architecture](docs/architecture.png?raw=true)

