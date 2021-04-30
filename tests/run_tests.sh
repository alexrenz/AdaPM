#!/bin/bash

if [ "$1" != "dont_compile" ]; then
    make tests
fi

workload="--quick"
if [ "$2" == "heavy" ]; then
    workload=""
fi

rm -rf logs
mkdir logs

######################
## run tests

# dynamic allocation
python tracker/dmlc_local.py -s 3 tests/test_dynamic_allocation | tee -a "logs/dynamic_allocation.log"
python tracker/dmlc_local.py -s 4 tests/test_dynamic_allocation replicate | tee -a "logs/dynamic_allocation_replicate.log"

# locality API
python tracker/dmlc_local.py -s 3 tests/test_locality_api | tee -a "logs/locality_api.log"

# variable length values
python tracker/dmlc_local.py -s 3 tests/test_variable_length_values | tee -a "logs/variable_length_values.log"

# many-key operations
python tracker/dmlc_local.py -s 4 tests/test_many_key_operations $workload | tee -a "logs/many_keys.log"

# many-key operations with replication
python tracker/dmlc_local.py -s 4 tests/test_many_key_operations $workload --replicate --rep.sm bg_tree | tee -a "logs/many_keys_replication_tree.log"
python tracker/dmlc_local.py -s 4 tests/test_many_key_operations $workload --replicate --rep.sm bg_butterfly | tee -a "logs/many_keys_replication_butterfly.log"


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

echo ""
echo "All tests passed"
exit 0

