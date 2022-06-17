![logo](docs/adaps.svg?raw=true) 

![Build on Latest Ubuntu](https://github.com/alexrenz/AdaPS/actions/workflows/latest-ubuntu.yml/badge.svg)
![Test bindings](https://github.com/alexrenz/AdaPS/actions/workflows/bindings.yml/badge.svg)
[![License](docs/apache2.svg?raw=true)](./LICENSE)

**AdaPS** is a fully adaptive parameter server (PS). AdaPS is efficient for many
machine learning tasks out of the box because it automatically adapts to the
underlying task. It adapts based on **intent signals**. I.e., the application
signals which parameters it intends to access in the near future. Based on these
signals, AdaPS decides automatically (i.e., without specific user input) and
adaptively (i.e., depending on the current situation) what to do and when to do
it. This makes AdaPS *efficient and easy to use*. We describe details in our
[paper on AdaPS (arXiv)](https://arxiv.org/abs/2206.00470).

The `main` branch of this repository contains the latest version of AdaPS.
Details on the experiments in the AdaPS paper
([arXiv](https://arxiv.org/abs/2206.00470)) can be found in
[docs/experiments.md](https://github.com/alexrenz/AdaPS/blob/review/docs/experiments.md).
The source code used in the paper is in branch
[`review`](https://github.com/alexrenz/AdaPS/tree/review/).

AdaPS is the successor of **Lapse** and **NuPS**. Lapse is the first PS
that supports dynamic parameter allocation, i.e., the ability to relocate
parameters among nodes during run time. Our paper on Lapse provides more
information ([PVLDB 13(12),
2020](https://www.vldb.org/pvldb/vol13/p1877-renz-wieland.pdf)). Details on the
experiments for this paper can be found in
[docs/experiments-vldb20.md](https://github.com/alexrenz/AdaPS/blob/vldb20/docs/experiments-vldb20.md).
The source code used in this paper is in branch
[`vldb20`](https://github.com/alexrenz/AdaPS/tree/vldb20/). NuPS is a novel
multi-technique PS that combines relocation and replication management
techniques, and supports sampling directly in the PS. Our paper on NuPS provides
more detail ([SIGMOD 22](https://dl.acm.org/doi/10.1145/3514221.3517860)). Details on the experiments of this paper can be found in
[docs/experiments-sigmod22.md](https://github.com/alexrenz/AdaPS/blob/sigmod22/docs/experiments-sigmod22.md).
The source code used in this paper is in branch
[`sigmod22`](https://github.com/alexrenz/AdaPS/tree/sigmod22/).


AdaPS provides bindings to PyTorch, see [bindings/](bindings/). 

The implementation of AdaPS is based on NuPS, Lapse, and [PS-Lite](https://github.com/dmlc/ps-lite). 

### Usage

AdaPS provides the following primitives to access parameters: 
- `Pull(keys)`: retrieve the values of a set of parameters (identified by keys)
- `Push(keys, updates)`: send (additive) updates for parameters

AdaPS provides the following primitives to signal intent:
- `Intent(keys, start, end)`: signal that the issuing worker intends to access `keys` between clock `start` (incl.) and `end` (excl.)
- `advanceClock()`: raise the clock of the issuing worker by 1

Additionally, AdaPS supports sampling access (as NuPS does) via the following primitives:
- `handle = PrepareSample(N)`: prepare a group of `N` samples
- `PullSample(handle)`: retrieve `N` samples from a prepared group

By default, the `Pull()`, `Push()`, and `PullSample()` primitives execute asynchronously. `Wait()` can be used to execute these primitives synchronously. For example: `Wait(Pull(keys))`.


A simple example:

```c++
  std::vector<uint64_t> keys = {1, 3, 5};
  std::vector<float> updates = {1, 1, 1};
  std::vector<float> vals;
  ps::KVWorker<float> kv;

  kv.Wait(kv.Pull(keys, &vals)); // access without intent
  kv.Wait(kv.Push(keys, updates));
  
  kv.Intent(keys, 1, 2);

  // ...

  kv.advanceClock(); // clock started at 0, so is at 1 now

  kv.Wait(kv.Pull(keys, &vals)); // access with intent
  kv.Wait(kv.Push(keys, updates)); // access with intent
  
  // sampling access
  auto h = kv.PrepareSample(3); // prepare a group of 20 samples
  kv.Wait(kv.PullSample(h, keys, vals)); // pull the 3 samples (keys.size() determines how many samples are pulled)
```

## Build

AdaPS requires a C++11 compiler such as `g++ >= 4.8` and boost for some the application examples. On Ubuntu >= 13.10, you
can install it by
```
sudo apt-get update && sudo apt-get install -y cmake build-essential git wget libboost-all-dev libzmq3-dev libprotobuf-dev protobuf-compiler libeigen3-dev
```

Then clone and build (without torch support)

```bash
git clone https://github.com/alexrenz/AdaPS
cd AdaPS
cmake -S . -B build    # (equivalent old style for CMake<3.14: mkdir build && cd build && cmake ..)
cmake --build build --target all -j
```

See [bindings/README.md](bindings/README.md) for how to build the **bindings**.

### CMake options

- Set `PROTOBUF_PATH` to link a specific protobuf installation (rather than relying on the system's default paths). E.g., we use this to build ABI-compatible PyTorch bindings (see [bindings/README.md](bindings/README.md)).
- Set `CMAKE_BUILD_TYPE=Debug` to build debug binaries.
- (Advanced) Set `PS_KEY_TYPE` to the data type that the PS should use as keys (default: `uint64_t`)
- (Advanced) Set `PS_LOCALITY_STATS` to collect detailed locality statistics during run time.
- (Advanced) Set `PS_TRACE_KEYS=1` to compile with key tracing support. Then set `--sys.trace.keys` and `--sys.stats.out` when starting an application on a cluster.

## Getting started

A very simple example can be found in [simple.cc](apps/simple.cc). To run it with one node and default parameters:

```bash
python tracker/dmlc_local.py -s 1 build/apps/simple
```

Or to run with 3 nodes and some specific parameters:

```bash
python tracker/dmlc_local.py -s 3 build/apps/simple -v 5 -i 10 -k 14 -t 4
```
Run `build/apps/simple --help` to see available parameters.


## Starting an application on a cluster

There are multiple start scripts. We commonly use the following ones:
- [tracker/dmlc_local.py](tracker/dmlc_local.py) to run on a local machine
- [tracker/dmlc_ssh.py](tracker/dmlc_ssh.py) to run on a cluster
To see more information, run `python tracker/dmlc_local.py --help`, for example.

The `-s` flag specifies how many processes/nodes to use. For example, `-s 4` uses 4 nodes. In each process, AdaPS starts one server thread and multiple worker threads. 

## Example Applications

You find example applications in the [apps/](apps/) directory and launch commands to locally run toy examples below. The toy datasets are in [apps/data/](apps/data/). 


### Knowledge Graph Embeddings
```
python tracker/dmlc_local.py -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 100 
```

### Word vectors
```
python tracker/dmlc_local.py -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --data_words 10000
```

### Matrix Factorization

```
python tracker/dmlc_local.py -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 2 --num_rows 6 --num_cols 4 --epochs 10
```

## Architecture

AdaPS starts one process per node. Within this process, worker threads access the parameter store directly. A parameter server thread handles requests by other nodes, and a synchronization manager thread triggers replica synchronization and intent communication.

![architecture](docs/architecture.png?raw=true)


### How to cite

The citation for AdaPS is as follows:

```bibtex
@misc{adaps,
  author = {Renz-Wieland, Alexander and Kieslinger, Andreas and Gericke, Robert and Gemulla, Rainer and Kaoudi, Zoi and Markl, Volker},
  title = {Good Intentions: Adaptive Parameter Servers via Intent Signaling},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2206.00470},
  url = {https://arxiv.org/abs/2206.00470},
}

```


If you wish to refer NuPS specifically, cite:

```bibtex

@inproceedings{nups,
  author = {Renz-Wieland, Alexander and Gemulla, Rainer and Kaoudi, Zoi and Markl, Volker},
  title = {NuPS: A Parameter Server for Machine Learning with Non-Uniform Parameter Access},
  year = {2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {To appear in the Proceedings of the 2022 ACM International Conference on Management of Data},
  location = {Chicago, Illinois, USA},
  series = {SIGMOD '22}
}
```


If you wish to refer Lapse specifically, cite:

```bibtex
@article{lapse,
  author = {Renz-Wieland, Alexander and Gemulla, Rainer and Zeuch, Steffen and Markl, Volker},
  title = {Dynamic Parameter Allocation in Parameter Servers},
  year = {2020},
  issue_date = {August 2020},
  publisher = {VLDB Endowment},
  volume = {13},
  number = {12},
  issn = {2150-8097},
  url = {https://doi.org/10.14778/3407790.3407796},
  doi = {10.14778/3407790.3407796},
  journal = {Proc. VLDB Endow.},
  month = jul,
  pages = {1877–1890},
  numpages = {14}
}
```
