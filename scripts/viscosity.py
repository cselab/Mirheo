#!/usr/bin/python

import sys
import numpy
import h5py
import re

force = float(sys.argv[1])
n = float(sys.argv[2])
f = h5py.File(sys.argv[3], 'r')

#n = numpy.average(f["density"][()])
vx = f["avgU"][()]
meanvx = abs(numpy.average(vx[:,0:vx.shape[1]/2, :, :]))

L = vx.shape[1]
visc = n*force*L*L / (48*meanvx)

print "Average velocity: ", meanvx
print "Viscosity: ", visc