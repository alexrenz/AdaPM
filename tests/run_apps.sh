#!/bin/bash

# DSGD matrix factorization
python tracker/dmlc_local.py -n 0 -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 2 --num_rows 6 --num_cols 4 --epochs 10  || exit 1

# Column-wise SGD matrix factorization (with replication)
python tracker/dmlc_local.py -n 0 -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 10 --num_rows 6 --num_cols 4 --epochs 10 --algorithm columnwise --replicate 2 --init_parameters 2 || exit 1

# Knowledge Graph Embeddings
python tracker/dmlc_local.py -n 0 -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 10 --eval_freq 2 || exit 1

# Word2Vec
python tracker/dmlc_local.py -n 0 -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --localize_pos 1 --data_words 10000 || exit 1
