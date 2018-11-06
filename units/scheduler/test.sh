#! /bin/sh

# TEST: scheduler
set -eu
udx.run ./scheduler/build/test_scheduler > status.out.txt 2>/dev/null
