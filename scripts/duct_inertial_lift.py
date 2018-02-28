#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import h5py

def trajectories(case, ceny, cenz, size):
	prefix = "/home/alexeedm/extern/daint/scratch/focusing_square_free/"
	
	files = sorted(glob.glob(prefix + case + "/pos/sphere*"))
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
	
	y = np.array([ float(x.split()[3]) for x in lines ])
	z = np.array([ float(x.split()[4]) for x in lines ])
		
	y = np.abs(y - ceny) / size
	z = np.abs(z - cenz) / size
	
	return y, z

def plottraj(case):
	z, y = trajectories(case, 49.0, 49.0, 47.0)
	p = plt.plot(y, z, lw=2)
	plt.plot(y[0], z[0], "o", ms=7, color=p[-1].get_color())

def mean_err_cut(vals):
	npvals = np.array(vals[5:]).astype(np.float)
	
	m = np.mean(npvals)
	v = np.var(npvals) / npvals.size
		
	return m,v

def dump_plots(ry, rz, _Fy, _Fz):
	
	Fy = np.array(_Fy)
	Fz = np.array(_Fz)
	
	lengths = np.sqrt(Fy*Fy + Fz*Fz)
	Fy = Fy / lengths
	Fz = Fz / lengths
	
	norm = matplotlib.colors.LogNorm()
	norm.autoscale(lengths)
	cm = plt.cm.rainbow
	
	sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	sm.set_array([])
	
#	plt.quiver(ry, rz, Fy, Fz, alpha=0.9, color=cm(norm(lengths)))
#	plt.colorbar(sm)
	plt.quiver(ry, rz, _Fy, _Fz, minlength=0, color="C0")

	plt.xlabel('y', fontsize=16)
	plt.ylabel('z', fontsize=16)
	#plt.grid()

	plt.axes().set_xlim([0.0, 0.8])
	plt.axes().set_ylim([0.0, 0.8])
	#plt.gca().invert_yaxis()

	plt.axes().set_aspect('equal', 'box', anchor='SW')
	plt.tight_layout()
	#plt.show()

	
norot=""#_norot"
prefix = "/home/alexeedm/extern/daint/scratch/focusing_square" + norot + "/"
#case = "case_5_0.1__80_20_1.5__"
case = "case_newcode_5_0.1__80_20_1.5__"

rho = 8.0
r = 5
R = 30

positionsy = np.linspace(0.0, 0.72, 10) + 0.04
positionsz = positionsy

#positionsy = np.linspace(0.0, 0.6, 9)
#positionsz = np.linspace(0.0, 0.1, 9) + 0.5
#
#positionsy = np.linspace(0.0, 0.1, 9) + 0.44
#positionsz = np.linspace(0.0, 0.1, 9) + 0.44

positionsy = np.linspace(0.0, 0.6, 31) + 0.04
positionsz = [0.04]

Fy = []
Fz = []
ry = []
rz = []

fig = plt.figure()

for posy in positionsy:
	for posz in positionsz:
		strposy = str(posy)
		strposz = str(posz)
		
		if strposy == "0.0":
			strposy = "0"
		
		full_folder = prefix + case + strposy + "x" + strposz
		
		files = sorted(glob.glob(full_folder + "/pinning_force/sphere*"))
		lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
							
		fy = [ x.split()[3] for x in lines ]
		fz = [ x.split()[4] for x in lines ]
		
		(my, vy) = mean_err_cut(fy)
		(mz, vz) = mean_err_cut(fz)
		
		ey = 3.0*math.sqrt(vy)
		ez = 3.0*math.sqrt(vz)
		
		print("%6f, %6f :    %6.2f  +- %4.2f    %6.2f +- %4.2f" % (posy, posz, my, ey, mz, ez))
		
#		if posy > posz:
#			my = 0
#			mz = 0
#			
#		my = 0
		
		Fy.append(my)
		Fz.append(mz)
		
		ry.append(posy)
		rz.append(posz)
	
		
#print Fy
#print Fz

dump_plots(ry, rz, Fy, Fz)

#plottraj("case" + norot + "_10_0.05__80_40_1.5")
#plottraj("case" + norot + "_10_0.05__80_40_1.5__1")
#plottraj("case" + norot + "_10_0.05__80_40_1.5__2")
#plottraj("case" + norot + "_10_0.05__80_40_1.5__3")

plt.show()
#fig.savefig("/home/alexeedm/udevicex/media/square_duct_comparison_colored.pdf", bbox_inches='tight', transparent=True)


