#!/bin/bash

# DSGD matrix factorization
python3 tracker/dmlc_local.py -n 0 -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 4 --num_rows 6 --num_cols 4 --epochs 10 --signal_intent_cols 1  || exit 1

# Column-wise SGD matrix factorization
python3 tracker/dmlc_local.py -n 0 -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 10 --num_rows 6 --num_cols 4 --epochs 10 --algorithm columnwise --init_parameters 2 --signal_intent_cols 3 || exit 1

# Knowledge Graph Embeddings
python3 tracker/dmlc_local.py -n 0 -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 10 --eval_freq 2 --init_parameters "uniform{-1/1}" || exit 1

# Word2Vec
python3 tracker/dmlc_local.py -n 0 -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --data_words 10000 || exit 1
