#! /bin/bash

usage() {
    cat <<EOF
wrapper to export tag to git for versioning udeviceX
usage: 
       ./tag_version.sh <VERSION>

where VERSION has a format X.Y.Z
      X: major version: increment after major feature changes
      Y: minor version: increment after each new feature
      Z: micro version: increment after bug fixes etc

run 
    git push origin master --tags

to push all tags on the repo
EOF
    exit 1
}

set -eu

if test $# -ne 0 && test "$1" = -h; then usage; fi

VERSION=$1; shift

git tag -a "v${VERSION}" -m "version v${VERSION}"

echo "created tag v${VERSION}"
echo "you can push the tags by invoking"
echo "git push origin master --tags"
