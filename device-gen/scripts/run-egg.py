#!/usr/bin/env python
'''
 *  Part of CTC/device-gen/scripts/run-egg.py
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

parser = argparse.ArgumentParser(description='Generates eggs: ',
                                 usage= './run-egg.py -r <nRows> -c <nColumns> -d <0|1>')
parser.add_argument('-r','--nRows', help='', required=True)
parser.add_argument('-c','--nColumns', help='', required=True)
#parser.add_argument('-d','--draw', help='values: 0 | 1', required=False, default=1)
args = vars(parser.parse_args())

marginZ = 5.0
marginY = 5.0 # responsible for side walls
ntimesInXdir = 2 # how many times to repeat
eggSize = [56, 32, 48 + 2*marginZ]

resolution = 1.4 #0.7
unitXRes = resolution*eggSize[0]
unitYRes = resolution*eggSize[1]
unitZRes = resolution*eggSize[2]
print("Unit grid size: %g %g %g\n"%(unitXRes, unitYRes, unitZRes))

nRows = int(args['nRows'])
nColumns = int(args['nColumns'])
#draw = int(args['draw']) == 1

angle = 1.7 * math.pi/180.0

os.system("../sdf-unit-egg/sdf-unit %d %d %g %g %s"%(unitXRes, unitYRes, eggSize[0], eggSize[1], "egg.dat"))

nRowsPerShift = int(math.ceil(eggSize[0] / (eggSize[1] * math.tan(angle))))
if (math.fabs(eggSize[0] / (eggSize[1] * math.tan(angle)) - nRowsPerShift) > 1e-1):
    print("ERROR: Suggest changing the angle")    
    exit()

padding = float(math.ceil(nRows * eggSize[1] * math.tan(angle)))

nUniqueRows = nRows
if (nRows > nRowsPerShift):
    nUniqueRows = nRowsPerShift
    padding = float(round(nRowsPerShift * eggSize[1] * math.tan(angle), 0))

print("nRowsPershift = %d, nUniqueRows = %d, Padding = %f"%(nRowsPerShift, nUniqueRows, padding))

# workaround
if (padding < 32):
    padding = 0
if (padding == 57):
    padding = 56
padding = padding + 8 # 8 change by hands if needed

print("Launching rows generation. Padding = %g"%(padding))
for i in range(nUniqueRows-1, -1, -1):
    os.system("../sdf-collage/sdf-collage %s %d %d %f %s"%("egg.dat", nColumns, 1, 0.0, "raw-row.dat"))
    xshift = i * 32.0 * math.tan(angle)
    print("Calling: ../sdf-shift/sdf-shift %s %g %g %s"%("raw-row.dat", xshift, padding, "row%d.dat"%(i)))
    os.system("../sdf-shift/sdf-shift %s %g %g %s"%("raw-row.dat", xshift, padding, "row%d.dat"%(i)))

with open("files.txt", 'w') as f:
    for i in range(nRows-1, -1, -1):
        j = i % nRowsPerShift
        f.write("row%d.dat\n"%(j))

os.system("../sdf-collage/sdf-collage files.txt 1 1 %f %s "%(marginY, "collage.dat"))

if (ntimesInXdir > 1): 
    os.system("mv collage.dat collage_temp.dat")
    os.system("../sdf-collage/sdf-collage collage_temp.dat %d %d 0.0 collage.dat"%(1, ntimesInXdir))

#if (draw):
#    os.system("../dat2hdf5/dat2hdf5  collage.dat 2d")
os.system("../2Dto3D/2Dto3D collage.dat %g %g %d %s"%(eggSize[2] - 2*marginZ, marginZ, unitZRes, "%dx%d.dat"%(nColumns, nRows)))
#if (draw):
#    os.system("../dat2hdf5/dat2hdf5  %dx%d.dat %dx%d"%(nColumns, nRows, nColumns, nRows))

os.system("rm row*.dat")
