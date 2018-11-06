#! /bin/sh

# TEST: interactions
set -eu
udx.run ./interaction/build/test_interaction > status.out.txt 2>/dev/null
