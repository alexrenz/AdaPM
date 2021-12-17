# Detailed experiment settings

This document provides details on the experiments of the following paper:

> A. Renz-Wieland, R. Gemulla, Z. Kaoudi, V. Markl.  
> *NuPS: A Parameter Server for Machine Learning with Non-Uniform Parameter Access*.  

The paper is available on [arXiv.org](https://arxiv.org/abs/2104.00501).

If you run into any problems, have questions, find bugs, or want to obtain any of our datasets, please do not hesitate to contact us. I (Alexander Renz-Wieland) plan to keep durable contact information available at [alexrw.org](https://www.alexrw.org). 


# Hyperparameters

## Knowledge graph embeddings
We ran random search to determine hyperparameters (learning rate, regularization
for entities, and regularization for relations): we ran 10 random settings
(generated using the random search function of
[LibKGE](https://github.com/uma-pi1/kge)) on a single node (32 threads) for 10
epochs and picked the setting with the best model after the 10th epoch
(ε=0.1869, λ<sub>entity</sub>= 2.8266e-10, λ<sub>relation</sub>=6.6268e-6).
Whenever relocation is used, workers pre-localize relocation-managed parameters
one data point before the parameter is accessed.

## Word vectors
We used the learning rate decay of the [original
implementation](https://github.com/tmikolov/word2vec). To determine the starting
step size α, we ran α ∈ {0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2} on a
single node for 1 epoch and picked the α that produced the best model (α =
0.00625) after 1 epoch. Whenever relocation is used, a worker pre-localizes
relocation-managed parameters when it reads a new sentence.

## Matrix factorization
To determine the initial step size ε and the regularization parameter λ, we ran
a grid search on ε ∈ {0.1, 0.01, 0.001, 0.0001} and λ ∈ {1, 0.1, 0.01, 0.001}.
We ran each configuration for 5 epochs, using 1 node and chose the combination
that produced the best test loss after 5 epochs (ε = 0.0001 and λ = 0.01). We
used the bold driver heuristic to automate step size selection after the first
epoch: step size was increased by 5\% if the training loss had decreased during
the epoch, and step size was decreased by 50\% if it had increased. Whenever
relocation is used, workers pre-localize relocation-managed parameters in groups
of 500 data points: when a new group of data points is reached, the parameters
for the next group are pre-localized.

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

# Launch commands

## General


To launch experiments, we used the [tracker/dmlc_ssh.py](/tracker/dmlc_ssh.py) launch script. For example:
```bash
python tracker/dmlc_ssh.py -H [PATH_TO_HOSTFILE] -s [1 or 8] [TASK_BINARY] [PROGRAM_OPTIONS]
```

The host file is a text file that contains one host name per line. The hosts have to be password-less SSH-able from all nodes. The `-s` Parameter controls the number of nodes.

We ran all experiments with 8 worker threads (`num_threads: 8`). For cluster runs, we used 8 nodes (`-s 8`), for the shared memory single node baseline, we used 1 node (`-s 1`).

To launch on the InfiniBand network, we added the option `-i ibs2` to the launcher.

We ran 3 independent runs of all experiments. To start each experiment with a distinct random starting point, we set `model_seed` to the values `23`, `343239821`, and `78974` in the three runs, respectively.

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
max_runtime: 6*60*60    # 6h

gamma_entity: 2.8265676000215333e-10 
gamma_relation: 6.626811716399634e-06 
eta: 0.1868603431873307

async_push: 1
read_partitioned_dataset: 0
rep.clip_updates: 0

init_parameters: 1
write_embeddings: [PATH WITH PLENTY AVAILABLE STORAGE]/[SOME UNIQUE ID].
write_every: 1
write_end_checkpoint: 0
eval_freq: -1
```

### System variants

For **NuPS (untuned)** (and **Reuse 16x**) we set
```
localize_parameters_ahead: 1 
replicate: 900
sampling.strategy: pool
sampling.reuse: 16
```
For **Reuse 64x** we used the same setting but set `sampling.reuse: 64`. For **Reuse with postponing (16x)** and **Reuse with postponing (64x)** we used the corresponding *Reuse* settings and additionally set `sampling.postpone: 1`.

For **NuPS** we set
```
localize_parameters_ahead: 1 
replicate: 900
sampling.strategy: onlylocal 
```

For **Lapse**, we set
```
localize_parameters_ahead: 1 
replicate: 0
sampling.strategy: preloc
```

For **Classic**, we set
```
localize_parameters_ahead: 0
replicate: 0
sampling.strategy: naive
```

In the Ablation, for **Relocation + Replication**, we set
```
localize_parameters_ahead: 1 
replicate: 900
sampling.strategy: preloc
```

For **Relocation + Sampling access management**, we set
```
localize_parameters_ahead: 1 
replicate: 0
sampling.strategy: pool
sampling.reuse: 16
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

num_iterations: 20
max_runtime: 6 * 60*60     #6h
starting_alpha: 0.00625
negative: 3

rep.clip_updates: 2
peek_ahead: 7
sync_push: 0
prep_context_ahead: 5

binary: 1
num_keys: 1880970
input_file: [PATH TO DATASET]/corpus.txt
vocab_retrieve: [PATH TO DATASET]/vocab.txt
output_file:    [PATH WITH PLENTY AVAILABLE SPACE]/[UNIQUE IDENTIFIER]
write_results:  1
```

### System variants

For **NuPS (untuned)** (and **Reuse 16x**) we set
```
replicate: 3272
localize_pos: 1 
sampling.strategy: pool
sampling.reuse: 16
```
For **Reuse 64x** we used the same setting but set `sampling.reuse: 64`. For **Reuse with postponing (16x)** and **Reuse with postponing (64x)** we used these *Reuse* settings and additionally set `sampling.postpone: 1`.

For **NuPS** we set
```
replicate: 209408
localize_pos: 1 
sampling.strategy: onlylocal 
```

For **Lapse** we set
```
replicate: 0
localize_pos: 1 
sampling.strategy: preloc
```

For **Classic** we set
```
replicate: 0
localize_pos: 0 
sampling.strategy: naive
```

In the Ablation, for **Relocation + Replication** we set
```
replicate: 3272
localize_pos: 1 
sampling.strategy: preloc

```

For **Relocation + Sampling access management** we set
```
replicate: 0
localize_pos: 1 
sampling.strategy: pool
sampling.reuse: 16
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
epochs: 500
eps: 0.0001
lambda: 0.01

enforce_random_keys: 1
compute_loss: 1
algorithm: columnwise
rep.clip_updates: 2
prelocalize_groupsize: 500

dataset: [PATH TO DATASET]
num_rows: 10000000
num_cols: 1000000
max_runtime: 3*60*60         # 3h
```


### System variants
For **NuPS (untuned)**, we set
```
replicate: 4889
prelocate_steps: 1
```

For **Lapse**, we set
```
replicate: 0
prelocate_steps: 1
```

For **Classic**, we set
```
replicate: 0
prelocate_steps: 0
```

## Other experiments
For the **Choice of Management Technique** experiments, we used the *NuPS (untuned)* settings, and varied `replicate` as explained in the paper.

For the **Effect of Replica Staleness** experiments, we used the *NuPS (untuned)* settings, and set `rep.syncs_per_sec` to the corresponding values (`0`, `0.2`, `1`, `5`, `25`, and `125`).
