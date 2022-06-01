#!/bin/bash

rm -rf logs
mkdir logs

if [ "$1" != "dont_compile" ]; then
    make tests -j || echo "Test compilation FAILED" > logs/compliation.log
fi

workload="--quick"
if [ "$2" == "heavy" ]; then
    workload=""
fi

######################
## run tests

# dynamic allocation
python3 tracker/dmlc_local.py -s 3 build/tests/test_dynamic_allocation | tee -a "logs/dynamic_allocation.log"

# locality API
python3 tracker/dmlc_local.py -s 3 build/tests/test_locality_api | tee -a "logs/locality_api.log"

# many-key operations
python3 tracker/dmlc_local.py -s 4 build/tests/test_many_key_operations $workload | tee -a "logs/many_keys.log"
python3 tracker/dmlc_local.py -s 4 build/tests/test_many_key_operations $workload --sys.techniques replication_only | tee -a "logs/many_keys_replication_only.log"
python3 tracker/dmlc_local.py -s 4 build/tests/test_many_key_operations $workload --sys.techniques relocation_only | tee -a "logs/many_keys_relocation_only.log"
python3 tracker/dmlc_local.py -s 4 build/tests/test_many_key_operations $workload --sys.location_caches 1 | tee -a "logs/many_keys_with_loc_caches.log"

# test set operation
python3 tracker/dmlc_local.py -s 4 build/tests/test_set_operation | tee -a "logs/set_operation.log"

# sampling support
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme naive | tee -a "logs/sampling_naive.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme naive --sampling.with_replacement 0 | tee -a "logs/sampling_naive_wor.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme preloc | tee -a "logs/sampling_preloc.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme preloc --sampling.with_replacement 0 | tee -a "logs/sampling_preloc_wor.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme pool --sampling.reuse 3 | tee -a "logs/sampling_pool.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme local --sampling.batch_size 1 | tee -a "logs/sampling_onlylocal.log"
python3 tracker/dmlc_local.py -s 3 build/tests/test_sampling --sampling.scheme local --sampling.batch_size 1 --sampling.with_replacement 0 | tee -a "logs/sampling_onlylocal_wor.log"

######################
## evaluate
echo ""
echo "Passed:"
echo "-----------------------------------------------------------------------"
grep -h 'PASSED' logs/*.log
echo "-----------------------------------------------------------------------"

echo ""
echo "Failed:"
echo "-----------------------------------------------------------------------"
grep -h 'FAILED' logs/*.log
echo "-----------------------------------------------------------------------"

# check result
anyfail=0
if (( `grep 'FAILED' logs/*.log | wc -l` > 0 )); then
    echo "At least one test has failed: FAIL"
    anyfail=1
fi

anynotpassed=0
echo ""
echo "Not passed:"
echo "-----------------------------------------------------------------------"
for f in logs/*.log; do
    if (( `grep 'PASSED' $f | wc -l` < 1 )); then
        echo "NOT PASSED: $(basename $f | cut -f 1 -d '.')"
        anynotpassed=$anynotpassed+1
    fi
done
echo "-----------------------------------------------------------------------"

if (( $anynotpassed > 0 )); then
    echo "At least one test has not passed. Thus: FAIL"
    exit 1
fi

if (( $anyfail > 0 )); then
    exit 1
fi

# check that all tests created a log file
num_tests=$(grep "^python" tests/run_tests.sh | wc -l)
num_logs=$(ls -l logs/*.log | wc -l)
echo ""
if (( $num_tests != $num_logs )); then
    echo "The number of tests ($num_tests) in the run_tests.sh script does not match the number of log files in logs/ ($num_logs). This is unexpected. Thus: FAIL"
    exit 1
else
    echo "The number of tests ($num_tests) in the run_tests.sh script matches the number of log files in logs/ ($num_logs)."
fi


echo ""
echo "All tests passed"
exit 0

