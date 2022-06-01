![logo](docs/adaps.svg?raw=true) 

[![license](docs/apache2.svg?raw=true)](./LICENSE)

AdaPS is a fully adaptive parameter server. This branch contains the source code of AdaPS that we used in the AdaPS paper ([available on arxiv soon](TODO)).

Details on the **experiment settings** of this paper can be found in [docs/experiments.md](docs/experiments.md).

The implementation of AdaPS is based on [PS-Lite](https://github.com/dmlc/ps-lite), Lapse, and NuPS.


### Long version

Please find the long version of the paper [here](long_version.pdf).

### Build

AdaPS requires a C++11 compiler such as `g++ >= 4.8` and boost. On Ubuntu >= 13.10, you
can install the dependencies via
```
sudo apt-get update && sudo apt-get install -y build-essential git libboost-all-dev
```

Then clone and build

```bash
git clone --branch review https://github.com/alexrenz/AdaPS/
cd AdaPS && make
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

There are multiple start scripts. For our experiments, we use the following ones:
- [tracker/dmlc_ssh.py](tracker/dmlc_ssh.py)
To see more information, run `python tracker/dmlc_local.py --help`, for example.

The `-s` flag specifies how many processes (i.e., nodes to use, e.g. `-s 4` uses 4 nodes. In each process, AdaPS starts one server thread and multiple worker threads. 

The exact launch commands for the AdaPS paper can be found in [docs/experiments.md](docs/experiments.md).
