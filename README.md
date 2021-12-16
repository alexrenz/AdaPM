![logo](docs/logo.svg?raw=true) 

![Build on Latest Ubuntu](https://github.com/alexrenz/NuPS/actions/workflows/latest-ubuntu.yml/badge.svg)
![Test bindings](https://github.com/alexrenz/NuPS/actions/workflows/bindings.yml/badge.svg)
[![GitHub license](docs/apache2.svg?raw=true)](./LICENSE)

NuPS is a general-purpose parameter server that aims to be efficient for many machine learning tasks, including tasks that exhibit non-uniform parameter access. NuPS integrates multiple parameter management techniques (replication and relocation) and allows applications to pick a suitable technique *per parameter*. This allows to manage both frequently and infrequently accessed parameters efficiently. Further, NuPS supports sampling directly via sampling primitives and sampling schemes that allow for a controlled quality--efficiency trade-off. Our paper on NuPS ([arXiv](https://arxiv.org/abs/2104.00501)) provides more details. 

The `main` branch of this repository contains the latest version of NuPS. Details on the experiments for the paper on NuPS ([arXiv](https://arxiv.org/abs/2104.00501), to appear in SIGMOD '22) can be found in [docs/experiments-sigmod22.md](https://github.com/alexrenz/NuPS/blob/sigmod22/docs/experiments-sigmod22.md). You find the source code used in the paper in branch [`sigmod22`](https://github.com/alexrenz/NuPS/tree/sigmod22/) and as release [`v2.0-sigmod22`](https://github.com/alexrenz/NuPS/releases/tag/v2.0-sigmod22). 

NuPS is the successor of **Lapse**, the first parameter server that supports dynamic parameter allocation, i.e., the ability to relocate parameters among nodes during run time. Our paper on Lapse provides more information ([PVLDB 13(12), 2020](https://www.vldb.org/pvldb/vol13/p1877-renz-wieland.pdf)). Details on the experiments for this paper can be found in [docs/experiments-vldb20.md](https://github.com/alexrenz/NuPS/blob/vldb20/docs/experiments-vldb20.md). You find the source code used in the paper in branch [`vldb20`](https://github.com/alexrenz/NuPS/tree/vldb20/) and as release [`v1.0-vldb20`](https://github.com/alexrenz/NuPS/releases/tag/v1.0-vldb20).

NuPS provides bindings to PyTorch, see [bindings/](bindings/). 

The implementation of NuPS is based on Lapse and [PS-Lite](https://github.com/dmlc/ps-lite). 

### Usage

NuPS provides the following primitives: 
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

NuPS requires a C++11 compiler such as `g++ >= 4.8` and boost for some the application examples. On Ubuntu >= 13.10, you
can install it by
```
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
```

Then clone and build

```bash
git clone https://github.com/alexrenz/NuPS
cd NuPS && make
```

See [bindings/README.md](bindings/README.md) for how to build the bindings.

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
python tracker/dmlc_local.py -s 4 build/tests/test_dynamic_allocation
```



### Starting an application

There are multiple start scripts. At the moment, we mostly use the following ones:
- [tracker/dmlc_local.py](tracker/dmlc_local.py) to run on a local machine
- [tracker/dmlc_ssh.py](tracker/dmlc_ssh.py) to run on a cluster
To see more information, run `python tracker/dmlc_local.py --help`, for example.

The `-s` flag specifies how many processes/nodes to use. For example, `-s 4` uses 4 nodes. In each process, NuPS starts one server thread and multiple worker threads. 

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

NuPS starts one process per node. Within this process, worker threads access the parameter store directly. A parameter server thread handles requests by other nodes and parameter relocations.

![architecture](docs/architecture.png?raw=true)


### How to cite
Please refer to the VLDB '20 paper if you use **Lapse**:

```bibtex
@article{10.14778/3407790.3407796,
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

And please refer to the SIGMOD '22 paper if you use **NuPS**:

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
