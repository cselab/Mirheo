#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

def trajectories(case, ceny, cenz, size):	
	files = sorted(glob.glob(case + "/pos/*.txt"))
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
	
	y = np.array([ float(x.split()[3]) for x in lines ])
	z = np.array([ float(x.split()[4]) for x in lines ])
	
	y = (y - ceny) / size
	z = (z - cenz) / size
	
	return y, z

def plottraj(case):
	z, y = trajectories(case, 33.25, 33.25, 31.25 )
	#p = plt.plot(y, z, lw=0.5)
	#plt.plot(y[0], z[0], "x", ms=3, color=p[-1].get_color())
	plt.plot(y[-1], z[-1], "o", ms=5)#, color=p[-1].get_color())

	
folder = "/home/alexeedm/extern/daint/scratch/focusing_square_free/"
name = "case_5_0.155__80_40_1.5__"

variants = sorted(glob.glob(folder + name + "*/"))

fig = plt.figure()
plt.axes().set_xlim([-1.05, 1.05])
plt.axes().set_ylim([-1.05, 1.05])

for case in variants:
	print case
	plottraj(case)
	
plt.show()
fig.savefig("/home/alexeedm/udevicex/media/square_duct_trajectories.pdf", bbox_inches='tight', transparent=True)


