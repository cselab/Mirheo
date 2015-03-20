#! /bin/bash
cd ../
for d in */ ; do 
    pushd $d
    make clean
    popd
done
