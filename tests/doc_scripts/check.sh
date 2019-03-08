#! /bin/sh

# TEST: docs.basic
cd doc_scripts/
rm -f exit.out.txt
ymr.run --runargs "-n 2" python basic.py > /dev/null
echo $? > exit.out.txt

# TEST: docs.hello
cd doc_scripts/
rm -f exit.out.txt
ymr.run --runargs "-n 1" python hello.py > /dev/null
echo $? > exit.out.txt

# TEST: docs.rest
cd doc_scripts/
rm -f exit.out.txt
ymr.run --runargs "-n 2" python rest.py > /dev/null
echo $? > exit.out.txt

# TEST: docs.walls
cd doc_scripts/
rm -f exit.out.txt
ymr.run --runargs "-n 2" python walls.py > /dev/null
echo $? > exit.out.txt
