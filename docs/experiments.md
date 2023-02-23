# Detailed experiment settings

This document provides details on the experiments of the AdaPM paper.

The KGE, WV, and MF applications are in [apps/](/apps/), the GNN and CTR applications are in [a separate repo](https://github.com/alexrenz/AdaPM-PyTorch-apps).

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

## Click-through-rate prediction and graph neural networks
We conducted no hyperparameter tuning for CTR and GNN. Instead, we used standard
hyperparameters.

# Datasets

Our datasets are available on kaggle:
- [Wikidata5M (knowledge graph embeddings)](https://www.kaggle.com/alexrenz/wikidata5m)
- [One billion words benchmark (word vectors)](https://www.kaggle.com/alexrenz/one-billion-words-benchmark)
- [10M x 1M, 1B matrix with zipf(1.1) skew (matrix factorization)](https://www.kaggle.com/alexrenz/syncthetic-zipf-1-1-matrix)
- [CTR](todo)
- [GNN](todo)

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

For **AdaPM** we additionally set
```
signal_intent_ahead: 1000 
sampling.scheme: "local"
```

For **Classic**, we set
```
signal_intent_ahead: 0
sampling_scheme: naive
```

For **Full replication**, we set
```
enforce_full_replication: 1
signal_intent_ahead: 0
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

For **AdaPM**, we set
```
signal_intent: 1
sampling_scheme: local
```

For **Classic**, we set
```
signal_intent: 0
sampling_scheme: naive
```

For **Full replication**, we set
```
enforce_full_replication: 1
signal_intent: 0
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
For **AdaPM**, we set
```
signal_intent_cols: 10000
```

For **Classic**, we set
```
signal_intent_cols: 0
signal_intent_rows: 0
```

For **Full replication**, we set
```
enforce_full_replication: 1
signal_intent_cols: 0
```

## Ablation variants

For **AdaPM without relocation**, we used the AdaPM settings for the task and
added `sys.techniques: "replication_only"`.

For **AdaPM without replication**, we used the AdaPM settings for the task and
added `sys.techniques: "relocation_only"`.

## NuPS

We used the [Ax](https://ax.dev/) script [nups_configs.py](nups_configs.py) to generate 5 configurations for NuPS:
```python
[
  {'num_replicas': 0.2198966453196176, 'ahead': 16}, # config 1
  {'num_replicas': 0.028787422221962775, 'ahead': 134}, # config 2
  {'num_replicas': 19.16504095301115, 'ahead': 1}, # config 3
  {'num_replicas': 43.543440004371234, 'ahead': 25}, # config 4
  {'num_replicas': 0.041529590348307506, 'ahead': 1}, # config 5
]
```

To obtain task-specific configurations from these 5 general configurations, we
calculated the NuPS heuristic for the number of replicas for each task (900 for
KGE, 3272 for WV, 755 for MF) and multiplied the general `num_replicas` factor
with the heuristic for the task (and converted the result to an integer). E.g.,
for config 1 in KGE, we instructed NuPS to replicate `int(900 *
0.2198966453196176) = 197` keys. I.e., we set `replicate: 197` for this KGE
configuration.

In detail, for KGE in NuPS, we set:
```
replicate: int(900 * num_replicas)
localize_parameters_ahead: ahead
sampling.strategy: onlylocal
```

For WV in NuPS, we set:
```
replicate: int(3272 * num_replicas)
prep_context_ahead: ahead
sampling.strategy: onlylocal
peek_ahead: 7   # adopted from the settings in the original NuPS paper
localize_pos = 1
```

For MF in NuPS, we set:
```
replicate: int(755 * num_replicas)
prelocalize_steps: ahead
```

Further, we set the following NuPS system parameters (for all tasks):
```
rep.syncs_per_sec = 1000
rep.clip_updates = 0
rep.average_updates = 0
sys.sender_thread = 1
enforce_random_keys = 1
```

## Experiments on scalability

For scalability experiments, we varied the number of servers (`-s N`). 

## Experiments on action timing

For the experiments on the effect of action timing, we varied `sys.time_intent_actions` between `0` and `1`, and set the signal offset (`signal_intent_ahead` in KGE, `read_sentences_ahead` in WV, and `signal_intent_cols` in MF) to 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, and 262144.

