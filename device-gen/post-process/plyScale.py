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

parser = argparse.ArgumentParser(description='Modifies ply file to be used for rendering.\n Example: ./plyScale.py -f input.ply -o out.ply -r 210 -cutX 10.0')
parser.add_argument('-f','--inputFile', help='Input file name', required=True)
parser.add_argument('-o','--outputFile', help='Output file name', required=True)
parser.add_argument('--lx', help='Desired size of bounding box (X axis)', required=False, default="0")
parser.add_argument('--ly', help='Desired size of bounding box (Y axis)', required=False, default="0")
parser.add_argument('--lz', help='Desired size of bounding box (Z axis)', required=False, default="0")
parser.add_argument('-r','--order', help='Reorder axis. By default 012, to swap x and z use 210', required=False, default="012")
helpStringForCut = 'Remove all the faces which are above specified value for %s. The origin is in the center of mass. If axis reordering was applied, axis are in new coordinates.'
parser.add_argument('--cutX', help=helpStringForCut%('X'), required=False, default="none")
parser.add_argument('--cutY', help=helpStringForCut%('Y'), required=False, default="none")
parser.add_argument('--cutZ', help=helpStringForCut%('Z'), required=False, default="none")
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

if (args['cutX'] != "none" or args['cutY'] != "none" or args['cutZ'] != "none"):
    print "Cut it!"
    cut = [1e6]*3
    if args['cutX'] != "none":
        cut[0] = float(args['cutX'])
    if args['cutY'] != "none":
        cut[1] = float(args['cutY'])
    if args['cutZ'] != "none":
        cut[2] = float(args['cutZ'])

    toDelete = list()
    newInx = [None]*len(vertices)
    j = 0
    for i in range(0, len(vertices)):
        if (vertices[i][0] > cut[0] or vertices[i][1] > cut[1] or vertices[i][2] > cut[2]):
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

