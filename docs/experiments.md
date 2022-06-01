# Detailed experiment settings

This document provides details on the experiments of the following paper:

> A. Renz-Wieland, A. Kieslinger, R. Gericke, R. Gemulla, Z. Kaoudi, V. Markl.  
> *Good Intentions: Adaptive Parameter Servers via Intent Signaling*.  

A preprint of this paper will be available soon on [arXiv.org](TODO).

<!-- If you run into any problems, have questions, find bugs, or want to obtain any of our datasets, please do not hesitate to contact us. I (Alexander Renz-Wieland) plan to keep durable contact information available at [alexrw.org](https://www.alexrw.org).  -->


# Hyperparameters

## Knowledge graph embeddings


We used random search to determine hyperparameters: we ran 20 quasi-random
settings (generated using the random search function of
[LibKGE](https://github.com/uma-pi1/kge)) on a single node (32 threads) for 10
epochs. We included the following hyperparameters with the following ranges in
the search:

- learning rate: range 0.0003--1.0, logarithmic scale
- regularization for entities: range 1.0e-20--0.1, logarithmic scale
- regularization for relations: range 1.0e-20--0.1, logarithmic scale
- dropout for entities: range 0.0--0.5
- dropout for relations: range 0--0.5
- parameter initialization: either normal with std in range 1.0e-05--1.0 (logarithmic scale), or uniform with a in range -1.0---1.0e-05 and b=1.0:

We picked the setting with the best filtered MRR after the 10th epoch:
- learning rate = 0.1868603431873307
- regularization for entities = 2.8265676000215333e-10
- regularization for relations = 6.626811716399634e-06
- dropout for entities = 0.4
- dropout for relations = 0.0
- parameter initialization = normal(0, 5.897e-05)


## Word vectors
To determine the starting step size α, we ran α ∈ {0.003125, 0.00625, 0.0125,
0.025, 0.05, 0.1, 0.2} on a single node for 1 epoch and picked the α that
produced the best model (α = 0.00625) after 1 epoch.

## Matrix factorization
To determine the initial step size ε and the regularization parameter λ, we ran
a grid search on ε ∈ {1, 0.1, 0.01, 0.001} and λ ∈ {10, 1, 0.1, 0.01, 0.001}. We
ran each configuration for 10 epochs on using 1 node with 32 threads and picked
the configuration that produced the best test loss after 10 epochs (ε = 0.1 and
λ = 1). We used the bold driver heuristic to automate step size selection after
the first epoch: step size was increased by 5\% if the training loss had
decreased during the epoch, and step size was decreased by 50\% if it had
increased.

# Datasets

Our datasets are available on kaggle:
- [Wikidata5M (knowledge graph embeddings)](https://www.kaggle.com/alexrenz/wikidata5m)
- [One billion words benchmark (word vectors)](https://www.kaggle.com/alexrenz/one-billion-words-benchmark)
- [10M x 1M, 1B matrix with zipf(1.1) skew (matrix factorization)](https://www.kaggle.com/alexrenz/syncthetic-zipf-1-1-matrix)

### Dataset generation (matrix factorization)
To generate the set of revealed cells for the matrix factorization dataset, we
sampled 1 billion random row and column indices from zipf(1.1, 10M) (for rows)
and zipf(1.1, 1M) (for columns) distributions (rejecting duplicate coordinates
until reaching 1 billion unique cells).

To generate values for the set of revealed cells, we randomly generated
rank-1000 factors, drawing values from `Normal(0, 1/sqrt(sqrt(rank)))` (so that
the entries in the product matrix have unit variance). We added `Normal(0, 0.1)`
noise to the revealed cells.

Finally, we randomly permuted columns and rows.


# Launch commands

## General


To launch experiments, we used the [tracker/dmlc_ssh.py](/tracker/dmlc_ssh.py) launch script. For example:
```bash
python tracker/dmlc_ssh.py -H [PATH_TO_HOSTFILE] -s [1 or 8] [TASK_BINARY] [PROGRAM_OPTIONS]
```

The host file is a text file that contains one host name per line. The hosts have to be password-less SSH-able from all nodes. The `-s` Parameter controls the number of nodes.

We ran all experiments with 32 worker threads (`num_threads: 32`). For cluster runs, we used 8 nodes (`-s 8`), for the shared memory single node baseline, we used 1 node (`-s 1`).

To launch on the InfiniBand network, we added the option `-i ibs2` to the launcher.

We ran 3 independent runs of all experiments. To start each experiment with a distinct random starting point, we set `model_seed` to the values `23`, `343239821`, and `78974` in the three runs, respectively. Unless specified otherwise, each run was given a time budget of 4 hours = 14400 seconds (`max_runtime: 14400`).

In the following, we provide all task-specific program options. Program options that are not mentioned explicitly were left at the default values defined in the source code (visible in the `process_program_options` of the corresponding source code file).

## Knowledge Graph Embeddings

The code for knowledge graph embeddings (KGE) is in [apps/knowledge_graph_embeddings.cc](/apps/knowledge_graph_embeddings.cc). Compile with `make apps/knowledge_graph_embeddings`. After successful compilation, the task binary can be found in `build/apps/knowledge_graph_embeddings`.

For all KGE experiments, we set
```
algorithm: ComplEx
embed_dim: 500

dataset: [PATH_TO_WIKIDATA_DATASET]
num_entities: 4818679
num_relations: 828

neg_ratio: 100
num_epochs: 100

gamma_entity: 2.8265676000215333e-10 
gamma_relation: 6.626811716399634e-06 
eta: 0.1868603431873307

init_parameters: "normal{0/5.897e-05}"
dropout_entity: 0.4
dropout_relation: 0
gamma_entity: 2.8265676000215333e-10
gamma_relation: 6.626811716399634e-06
eta: 0.1868603431873307

async_push: 1

write_embeddings: [PATH WITH PLENTY AVAILABLE STORAGE]/[SOME UNIQUE ID].
write_every: 1
write_end_checkpoint: 0
eval_freq: -1
```

### System variants

For **AdaPS** we additionally set
```
signal_intent_ahead: 1000 
sampling.scheme: "local"
```

For **Classic**, we set
```
signal_intent: 0
sampling_scheme: naive
```



### Measuring model quality

The implementation writes model snapshots to the path specified in `write_embeddings`. We then evaluated these snapshots with [LibKGE](https://github.com/uma-pi1/kge). To do so, [we extended LibKGE](https://github.com/alexrenz/kge) such that it can import binary dumps of our format. We ran LibKGE to create a checkpoint for the model that we use (Complex-500 on the Wikidata5M dataset). To evaluate a dumped snapshot, we then ran
```
local/bin/kge valid [LIBKGE CHECKPOINT FOR WIKIDATA5M] --user.read_binary [DUMPED SNAPSHOT] 
```

## Word Vectors

The code for word vectors (WV) is in [apps/word2vec.cc](/apps/word2vec.cc). Compile with `make apps/word2vec`. After successful compilation, the task binary can be found in `build/apps/word2vec`.

For all WV experiments, we set
```
window: 5
min_count: 1
subsample: 1e-2
embed_dim: 1000

num_iterations: 40
starting_alpha: 0.00625
negative: 3

sync_push: 0
read_sentences_ahead: 2000

binary: 1
num_keys: 1880970
input_file: [PATH TO DATASET]/corpus.txt
vocab_retrieve: [PATH TO DATASET]/vocab.txt
output_file:    [PATH WITH PLENTY AVAILABLE SPACE]/[UNIQUE IDENTIFIER]
write_results:  1
```

### System variants

For **AdaPS**, we set
```
signal_intent: 1
sampling_scheme: local
```

For **Classic**, we set
```
signal_intent: 0
sampling_scheme: naive
```

### Measuring model quality

We used the `compute-accuracy` program of the [original Word2Vec implementation](https://github.com/tmikolov/word2vec) to evaluate WV model quality. To do so, we ran 
```
compute-accuracy [SNAPSHOT] < questions-words.txt
```
in the Word2Vec directory. 

## Matrix Factorization

The code for matrix factorization (MF) is in [apps/matrix_factorization.cc](/apps/matrix_factorization.cc). Compile with `make apps/matrix_factorization`. After successful compilation, the task binary can be found in `build/apps/matrix_factorization`. 

For all MF experiments, we set

```
rank: 1000
epochs: 300
algorithm: columnwise
init_parameters: 2
compute_loss: 1

eps: 0.1
lambda: 1

dataset: [PATH TO DATASET]
num_rows: 10000000
num_cols: 1000000
```


### System variants
For **AdaPS**, we set
```
signal_intent_cols: 10000
```

For **Classic**, we set
```
signal_intent_cols: 0
signal_intent_rows: 0
```


## Ablation variants

For **AdaPS, replicate-on-intent**, we used the AdaPS settings for the task and
added `sys.techniques: "replication_only"`.

For **AdaPS, relocate-on-intent**, we used the AdaPS settings for the task and
added `sys.techniques: "relocation_only"`.


## Experiments on scalability

For scalability experiments, we varied the number of servers (`-s N`). 

## Experiments on action timing

For the experiments on the effect of action timing, we varied `sys.time_intent_actions` between `0` and `1`, and set the signal offset (`signal_intent_ahead` in KGE, `read_sentences_ahead` in WV, and `signal_intent_cols` in MF) to 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, and 262144.

