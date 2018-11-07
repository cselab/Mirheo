#! /bin/sh

#TODO: stronger test

# TEST: scheduler
set -eu
udx.run ./scheduler/build/test_scheduler 2>/dev/null | sort > status.out.txt
