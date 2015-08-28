#!/usr/bin/env python

'''
 *  Part of CTC/device-gen/post-processing/plyScale.py
 *
 *  Created and authored by Kirill Lykov on 2015-08-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
'''

from plyfile import PlyData, PlyElement
import argparse
import copy
import numpy

def computeExtent(vertices):
    extentMax = [-10e6] * 3
    extentMin = [10e6] * 3
    for i in range(0, len(vertices)):
        vi = vertices[i]
        for dim in range(0, 3):
            extentMax[dim] = max(extentMax[dim], vi[dim])
            extentMin[dim] = min(extentMin[dim], vi[dim])
    
    origOrigin = [(extentMax[i] + extentMin[i])/2.0 for i in range(0, 3)]
    origExtent = [extentMax[i] - extentMin[i] for i in range(0, 3)]
    return (origOrigin, origExtent)

parser = argparse.ArgumentParser(description='Scales ply file.\n Example: ./plyScale.py -f input.ply -o out.ply -x 150 -y 40 -z 48')
parser.add_argument('-f','--inputFile', help='Input file name', required=True)
parser.add_argument('-o','--outputFile', help='Output file name', required=True)
parser.add_argument('-x','--lx', help='', required=False, default="0")
parser.add_argument('-y','--ly', help='', required=False, default="0")
parser.add_argument('-z','--lz', help='', required=False, default="0")
parser.add_argument('-r','--order', help='values are 0-2', required=False, default="012")
parser.add_argument('-c','--cut', help='values are 0-2', required=False, default="none")
args = vars(parser.parse_args())

desiredBox = [float(args['lx']), float(args['ly']), float(args['lz'])]

plydata = PlyData.read(args['inputFile'])
vertices = plydata['vertex'].data

# swap coords
order = args['order']
if (order != "012"):
    idx =  [int(order[i]) for i in range(0, len(order))]
    assert(len(idx) == 3)
    print "Swapping axis!"
    for i in range(0, len(vertices)):
        v = copy.deepcopy(vertices[i])
        for dim in range(0, 3):
            vertices[i][dim] = v[ idx[dim] ]


# Current box
(origOrigin, origExtent) = computeExtent(vertices)
for dim in range(0, 3):
    if desiredBox[dim] == 0:
        desiredBox[dim] = origExtent[dim]    
print ("Extent is (%f, %f, %f). Center is (%f, %f, %f)."%(origExtent[0], origExtent[1], origExtent[2], 
                                                          origOrigin[0], origOrigin[1], origOrigin[2]))
for i in range(0, len(vertices)):
    for dim in range(0, 3):
        vertices[i][dim] -= origOrigin[dim]
        vertices[i][dim] *= desiredBox[dim]/origExtent[dim]

if (args['cut'] != "none"):
    print "Cut it!"
    toDelete = list()
    dim = int(args['cut'])
    newInx = [None]*len(vertices)
    j = 0
    for i in range(0, len(vertices)):
        if (vertices[i][dim] > 0.0):
            toDelete.append(i)
        else:
            newInx[i] = j
            j += 1
            
    plydata['vertex'].data = numpy.delete(plydata['vertex'].data, toDelete, axis=0)
    # remove polygons containing these vertices
    setVertToDel = set(toDelete)
    faces = plydata['face'].data
    facesToDelete = list()
    for i in range(0, len(faces)):
        curr = set(faces[i][0])
        common = curr & setVertToDel
        if (common):
            facesToDelete.append(i)
    plydata['face'].data = numpy.delete(plydata['face'].data, facesToDelete, axis=0)
    
    #update vertices in polygons
    faces = plydata['face'].data
    for i in range(0, len(faces)):
        f = faces[i][0]
        for i in range(0, len(f)):
            f[i] = newInx[ f[i] ]

(finalOrigin, finalExtent) = computeExtent(vertices)
print ("Extent is (%f, %f, %f). Center is (%f, %f, %f)."%(finalExtent[0], finalExtent[1], finalExtent[2], 
                                                          finalOrigin[0], finalOrigin[1], finalOrigin[2]))


plydata.write(args['outputFile'])
