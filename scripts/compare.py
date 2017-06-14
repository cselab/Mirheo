#!/usr/bin/python

import sys
import numpy
import h5py
import re

f = h5py.File(sys.argv[1], 'r')

vx = f["avgU"][()]
vy = f["avgV"][()]
meanvx = numpy.average(vx, 0)
meanvy = numpy.average(vy, 0)
vxy = numpy.empty(meanvx.size + meanvy.size, dtype=meanvx.dtype)
vxy = vxy.reshape(meanvx.shape[0]*2, meanvx.shape[1], meanvx.shape[2])

vxy[0::2] = meanvx
vxy[1::2] = meanvy
vxy = vxy.reshape(vxy.shape[0], vxy.shape[1])

analytic = numpy.transpose(numpy.genfromtxt(sys.argv[2], delimiter=' '))

avgMy = numpy.mean(vxy[0::2])
avgAn = numpy.mean(analytic[0::2])

print avgAn, avgMy

diff = numpy.fabs(analytic - vxy)
diff2 = diff * diff

Linf = numpy.amax(diff)
L2 = numpy.sqrt(numpy.mean(diff2))

diff_rel = diff / ( numpy.maximum(numpy.fabs(analytic), 1.0e-5 * numpy.ones(analytic.shape)) ) 
diff2_rel = diff_rel * diff_rel

Linf_rel = numpy.amax(diff_rel)
L2_rel = numpy.sqrt(numpy.mean(diff2_rel))

numpy.savetxt("diff.csv", analytic[0::2] - vxy[0::2], delimiter=",")
numpy.savetxt("an.csv", analytic[0::2], delimiter=",")
numpy.savetxt("my.csv", vxy[0::2], delimiter=",")
print "Linf = ", Linf, "L2 = ", L2, "Linf_rel = ",  Linf_rel, "L2_rel = ", L2_rel

