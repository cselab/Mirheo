#!/bin/bash

set -eu

log() {
    echo "[mirheo] $@"
}

use_double=`mir.run -n 1 python -m mirheo compile_opt useDouble`

if [ $use_double = 1 ] ; then
    log "Testing for double precision."
    atest_dir=test_data_double
else
    log "Testing for single precision."
    atest_dir=test_data
fi

test_log=atest.log
log "Testing log will be saved to $test_log."

ATEST_DIR=$atest_dir atest `find . \( -name "*.py" -o -name "*.sh" \)` 2>&1 | tee $test_log
