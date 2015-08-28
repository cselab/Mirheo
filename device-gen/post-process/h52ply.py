#!/usr/bin/env /Applications/paraview.app/Contents/bin/pvpython 

'''
 *  Part of CTC/device-gen/post-processing/h52ply.py
 *
 *  Created and authored by Kirill Lykov on 2015-08-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
'''

import argparse
import os
from paraview.simple import *

print("h52ply started")
parser = argparse.ArgumentParser(description='Transforms h5 which has xmf to ply using Paraview python lib.',
                                 usage= './h52ply.py -i <input.xmf> -o <output.ply>')
parser.add_argument('-i','--inputFile', help='XMF', required=True)
parser.add_argument('-o','--outputFile', help='PLY', required=True)
args = vars(parser.parse_args())

# paraview wants to have absolute path
fullPath = os.path.dirname(os.path.abspath(args['inputFile'])) + '/'
print fullPath

a13x59xmf = XDMFReader(FileNames=[fullPath + args['inputFile']])
#a13x59xmf.GridStatus = ['Grid_26']

contour1 = Contour(Input=a13x59xmf)
contour1.Isosurfaces = [0.0]

# save data
SaveData(fullPath + args['outputFile'], proxy=contour1)

