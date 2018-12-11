#! /bin/sh

# TEST: docs.basic
cd doc_scripts/
rm -f exit.out.txt
ymr.run --runargs "-n 2" python basic.py > /dev/null
echo $? > exit.out.txt
