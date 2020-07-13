# Detailed experiment settings

This document gives details about the experiments that were run in the Lapse code base and were reported in the following paper:

> A. Renz-Wieland, R. Gemulla, S. Zeuch, V. Markl.  
> *Dynamic Parameter Allocation in Parameter Servers*.  
> PVLDB, 13(11): 1877-1890, 2020

The paper can be downloaded from the [PVLDB Volume 13 page](http://www.vldb.org/pvldb/vol13.html) ([PDF (TODO)](TODO)). A slightly longer version of this paper can be obtained from [arXiv](https://arxiv.org/abs/2002.00655). 

If you run into any problems, have questions, find bugs, or want to obtain any of our datasets, please do not hesitate to contact us. I (Alexander Renz-Wieland) plan to keep durable contact information available at [alexrw.org](https://www.alexrw.org). 

## General

We ran 3 independent runs of all our experiments. We ran each setting with `1`, `2`, `4`, and `8` nodes. 

To launch experiments, we used the [tracker/dmlc_ssh.py](/tracker/dmlc_ssh.py) launch script. For example:
```bash
python tracker/dmlc_ssh.py -H [PATH_TO_HOSTFILE] -s [1-8] [TASK_BINARY] [PROGRAM_OPTIONS]
```

The host file is a text file that contains one host name per line. The hosts have to be password-less SSH-able from all nodes. The `-s` Parameter controls the number of nodes.

In all experiments, we set 
```
num_threads = 4
```

In the following, we provide all task-specific program options. Program options that are not mentioned explicitly were left at the default values defined in the source code (visible in the `process_program_options` of the corresponding source code file).


## Matrix Factorization

The code for matrix factorization (MF) is in [apps/matrix_factorization.cc](/apps/matrix_factorization.cc). Compile with `make apps/matrix_factorization`. After successful compilation, the task binary can be found in `build/apps/matrix_factorization`. 

For all MF experiments, we set

```
rank = 100
epochs = 1
lambda = 0
```

### Datasets
To generate the MF datasets, we used the [`generateSyntheticData`](https://github.com/uma-pi1/DSGDpp/blob/master/tools/generateSyntheticData.cc) utility in the DSGDpp repository. We set `nnzSmall = 1000000000` and `r = 100` (in the source code) for both datasets. 

To generate the **3.4m x 3m** matrix, we additionally set `size1 = 3400000` and `size2 = 3000000`. To run MF on the dataset, we set
```
dataset = [PATH_TO_1B_SQ_DATASET_FOLDER]
num_keys = 6400000
```

To generate the **10m x 1m** matrix, we additionally set `size1 = 10000000` and `size2 = 1000000`. To run MF on the dataset, we set
```
dataset = [PATH_TO_1B_RECT_DATASET_FOLDER]
num_keys = 11000000
```

### System variants
For **Lapse**, we set
```
localize = 1
```

For **Classic PS with fast local access (in Lapse)**, we set
```
localize = 0
enforce_random_keys = 1
```

### Example
As example, to launch MF in **Lapse** on the **10m x 1m** dataset, using **8 nodes**, we ran:

```bash
python tracker/dmlc_ssh.py -H ~/workers -s 8 build/apps/matrix_factorization --num_threads 4 --rank 100 --epochs 1 --lambda 0 --dataset /data/mf/1b_rect/ --num_keys 11000000 --localize 1
```

## Knowledge Graph Embeddings

The code for knowledge graph embeddings (KGE) is in [apps/knowledge_graph_embeddings.cc](/apps/knowledge_graph_embeddings.cc). Compile with `make apps/knowledge_graph_embeddings`. After successful compilation, the task binary can be found in `build/apps/knowledge_graph_embeddings`.

For all KGE experiments, we set
```
dataset = [PATH_TO_DBPEDIA500_DATASET]
num_entities = 490598
num_relations = 573
async_push = 0
read_partitioned_dataset = 1
neg_ratio = 10
num_epochs = 1
eval_freq = -1
```

### Models
We ran three models. For **ComplEx-Small**, we set:
```
algorithm = "ComplEx"
embed_dim = 100
```

For **ComplEx-Large**, we set:
```
algorithm = "ComplEx"
embed_dim = 4000
```

For **RESCAL-Large**, we set:
```
algorithm = "RESCAL"
embed_dim = 100
```

### System variants

For **Lapse**, we set
```
localize_entities_ahead = 1
localize_relations = 1
```

For **Classic PS with fast local access (in Lapse)**, we set
```
localize_entities_ahead = 0
localize_relations = 0
enforce_random_keys = 1
```

For **Lapse, only data clustering**, we set
```
localize_entities_ahead = 0
localize_relations = 1
```

## Word Vectors

The code for word vectors (WV) is in [apps/word2vec.cc](/apps/word2vec.cc). Compile with `make apps/word2vec`. After successful compilation, the task binary can be found in `build/apps/word2vec`.

For all WV experiments, we set
```
embed_dim = 1000
num_iterations = 1

negative = 25
window = 5
subsample=1e-5
min_count = 2

num_keys = 1102470
sync_push = 0
write_results = 1
binary = 1
input_file = /PATH_TO_DATASET/corpus.txt
vocab_retrieve = /PATH_TO_DATASET/vocab.txt
```

### System variants

For **Lapse**, we set
```
localize_pos = 1
localize_neg = 1
only_local_neg = 1
nls = 4000
localize_next = 100
```

For **Classic PS with fast local access (in Lapse)**, we set 
```
localize_pos = 0
localize_neg = 0
only_local_neg = 0
```

