#!/bin/bash
# main script of travis

if [ ${TASK} == "lint" ]; then
    make lint || exit 1
fi

if [ ${TASK} == "build" ]; then
    make DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1
fi

if [ ${TASK} == "test" ]; then
    ## compile
    make tests DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1
    make apps DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit 1

    ## run tests
    tests/run_tests.sh dont_compile || exit 1

    ## run demo apps
    tests/run_apps.sh || exit 1
fi
