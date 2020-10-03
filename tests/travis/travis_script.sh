#!/bin/bash
# main script of travis

if [ ${TASK} == "lint" ]; then
    make lint || exit 1
fi

if [ ${TASK} == "build" ]; then
    make DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1
fi

if [ ${TASK} == "test" ]; then
    make tests DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1
    cd tests
    # single-worker tests
    tests=( test_connection test_dynamic_allocation test_locality_api )
    for test in "${tests[@]}"
    do
        find $test -type f -executable -exec ./repeat.sh 4 ./local.sh 3 0 ./{} \;
    done
    # multi-workers test
    # multi_workers_tests=( test_kv_app_multi_workers )
    # for test in "${multi_workers_tests[@]}"
    # do
    #     find $test -type f -executable -exec ./repeat.sh 4 ./local_multi_workers.sh 2 2 ./{} \;
    # done



    ######
    # run applications on demo datasets

    cd ..

    make apps DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1

    python tracker/dmlc_local.py -n 0 -s 2  build/apps/matrix_factorization --dataset apps/data/mf/ -r 2 --num_keys 12 --epochs 10  || exit 1

    python tracker/dmlc_local.py -n 0 -s 2 build/apps/knowledge_graph_embeddings --dataset apps/data/kge/ --num_entities 280 --num_relations 112 --num_epochs 4 --embed_dim 100 --eval_freq 2 || exit 1

    python tracker/dmlc_local.py -n 0 -s 2 build/apps/word2vec --num_threads 2 --negative 2 --binary 1 --num_keys 4970 --embed_dim 10  --input_file apps/data/lm/small.txt --num_iterations 4 --window 2 --localize_pos 1 --localize_neg 1 --data_words 10000 || exit 1
fi
