#! /bin/sh

# TEST: serializer
set -eu
udx.run ./serializer/build/serializer > status.out.txt 2>/dev/null
