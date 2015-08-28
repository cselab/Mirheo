#!/usr/bin/env python
'''
 *  Part of CTC/device-gen/scripts/run-parab.py
 *
 *  Created and authored by Kirill Lykov on 2015-08-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
'''

import os
import math
import argparse

parser = argparse.ArgumentParser(description='Generates parabolic funnel obstacles: ',
                                 usage= './run-parab.py -r <nRows -c <nColumns>')
parser.add_argument('-r','--nRows', help='', required=True)
parser.add_argument('-c','--nColumns', help='', required=True)
parser.add_argument('-d','--draw', help='values: 0 | 1', required=False, default=1)
args = vars(parser.parse_args())

nRows = int(args['nRows'])
nColumns = int(args['nColumns'])
draw = int(args['draw']) == 1

with open("files.txt", 'w') as f:
    for i in range(3, nRows + 3):
        f.write("r%d.dat\n"%(i))

unitXRes=24/2
unitYRes=96/2
unitZRes=128/2

cleftWidth = 0.5
zMargin = 4.0
zSize = 128.0

for i in range(3, nRows+3):
    os.system("../sdf-unit-par/sdf-unit %d %d %f %f %f gap%d.dat"%(unitXRes, unitYRes, 24.0, 96.0, cleftWidth*i, i))

for i in range(3, nRows+3):
    os.system("../sdf-collage/sdf-collage gap%d.dat %d %d %f r%d.dat"%(i, nColumns, 1, 0.0, i))

os.system("../sdf-collage/sdf-collage files.txt 1 1 0.0 collage.dat")
if draw == 1:
    os.system("../dat2hdf5/dat2hdf5 collage.dat 2d")
outFile = "%dx%d.dat"%(nRows, nColumns)
os.system("../2Dto3D/2Dto3D collage.dat %f %f %d %s"%(zSize - 2*zMargin, zMargin, unitZRes, outFile))
if draw == 1:
    os.system("../dat2hdf5/dat2hdf5 %s 3d"%(outFile))

os.system("rm row*.dat")
#os.system("tar -czvf %dx%d.tar.gz %s"%(nRows, nColumns, outFile))
