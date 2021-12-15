[![license](docs/apache2.svg?raw=true)](./LICENSE)

<meta name="robots" content="noindex">

NuPS is a parameter server for machine learning with non-uniform parameter access. 

This branch contains the version of NuPS that was used in the SIGMOD '22 paper on NuPS ([author version](https://arxiv.org/abs/2104.00501)).
Details on the **experiment settings** of this paper can be found in [docs/experiments-sigmod22.md](docs/experiments-sigmod22.md). The latest version of NuPS can be found in the `main` branch of this repository.

The implementation of NuPS is based on [PS-Lite](https://github.com/dmlc/ps-lite) and Lapse.

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
git clone --branch sigmod22 https://github.com/alexrenz/NuPS/
cd NuPS && make
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


### Starting an application

There are multiple start scripts. At the moment, we mostly use the following ones:
- [tracker/dmlc_local.py](tracker/dmlc_local.py) to run on a local machine
- [tracker/dmlc_ssh.py](tracker/dmlc_ssh.py) to run on a cluster
To see more information, run `python tracker/dmlc_local.py --help`, for example.

The `-s` flag specifies how many processes (i.e., nodes to use, e.g. `-s 4` uses 4 nodes. In each process, NuPS starts one server thread and multiple worker threads. 
