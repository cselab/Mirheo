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
import matplotlib.image as mpimg

def trajectories(case, ceny, cenz, size):	
	files = sorted(glob.glob(case + "/pos/*.txt"))
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
	
	try:
		y = np.array([ float(x.split()[3]) for x in lines ])[0:]
		z = np.array([ float(x.split()[4]) for x in lines ])[0:]
	except:
		print "Error reading"
		return [], []
	
	y = np.abs(y - ceny) / size
	z = np.abs(z - cenz) / size
	
	return y, z

def plottraj(case):
	z, y = trajectories(case, 52, 52, 50.0 )
	if len(z) < 1:
		return
	
	plt.plot(y, z, lw=0.1, zorder=1, color="C0")
	plt.plot(y[0], z[0],   "x", ms=2, color="C0", zorder=1)
	plt.plot(y[-1], z[-1], "o", ms=4, color="C3", zorder=10)


fig = plt.figure()

# Duct
folder = "/home/alexeedm/extern/daint/scratch/focusing_square_free/"
name = "case_8_0.04__80_40_1.5__"
reference = mpimg.imread("/home/alexeedm/Pictures/choi_fig2f.png")
plt.imshow(reference, extent=(-1,1, -1,1), zorder=0)


variants = sorted(glob.glob(folder + name + "*/"))

plt.axes().set_xlim([-1.1, 1.1])
plt.axes().set_ylim([-1.1, 1.1])



for case in variants:
	print case
	plottraj(case)
	
plt.show()
fig.savefig("/home/alexeedm/udevicex/media/square_duct_trajectories.pdf", bbox_inches='tight', transparent=True)


