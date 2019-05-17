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

# TEST: docs.membrane
cd doc_scripts/
cp ../../data/rbc_mesh.off .
rm -f exit.out.txt
ymr.run --runargs "-n 2" python membrane.py > /dev/null
echo $? > exit.out.txt

# TEST: docs.membrane.solvents
cd doc_scripts/
cp ../../data/rbc_mesh.off .
rm -f exit.out.txt
ymr.run --runargs "-n 2" python membranes_solvents.py > /dev/null
echo $? > exit.out.txt

# TEST: docs.rigid.generate
cd doc_scripts/
rm -f exit.out.txt
cp ../../data/sphere_mesh.off .
ymr.run --runargs "-n 1" python generate_frozen_rigid.py  > /dev/null
ymr.run --runargs "-n 2" python rigid_suspension.py  > /dev/null
echo $? > exit.out.txt
