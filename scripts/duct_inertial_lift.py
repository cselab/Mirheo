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

def trajectories(case, ceny, cenz, size):
	prefix = "/home/alexeedm/extern/daint/scratch/focusing_square_soft_free/"
	
	files = sorted(glob.glob(prefix + case + "/pos/*.txt"))
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
	
	y = np.array([ float(x.split()[3]) for x in lines ])
	z = np.array([ float(x.split()[4]) for x in lines ])
		
	y = np.abs(y - ceny) / size
	z = np.abs(z - cenz) / size
	
	return y, z

def plottraj(case):
	z, y = trajectories(case, 50.0, 50.0, 47.0)
	p = plt.plot(y, z, lw=2)
	plt.plot(y[0], z[0], "o", ms=5, color=p[-1].get_color())

def mean_err_cut(vals):
	npvals = np.array(vals[10:]).astype(np.float)
	
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
	plt.quiver(ry, rz, _Fy, _Fz, minlength=0)#, scale=1000, width=0.004)

	plt.xlabel('y', fontsize=16)
	plt.ylabel('z', fontsize=16)
	#plt.grid()

	plt.axes().set_xlim([0.0, 0.72])
	plt.axes().set_ylim([0.0, 0.72])
	plt.gca().invert_yaxis()

	plt.axes().set_aspect('equal', 'box', anchor='SW')
	plt.tight_layout()
	#plt.show()

	
prefix = "/home/alexeedm/extern/daint/scratch/focusing_square_soft/"
case = "case_0.1_20.0__80_20_1.5__"
#case = "case_0.038_20.0__80_6_1.5__"


#prefix = "/home/alexeedm/extern/daint/scratch/focusing_square/"
#case = "case_newcode_5_0.038__80_6_1.5__"
#case = "case_newcode_2.5_0.038__80_6_1.5__"
#case = "case_big_5_0.0027__80_6_1.5__"


prefix = "/home/alexeedm/extern/daint/project/alexeedm/focusing_square/"
case = "case_5_0.1__80_20_1.5__"


rho = 8.0
r = 5
R = 30

positionsy = np.linspace(0.0, 0.72, 10) + 0.04
positionsz = positionsy

#positionsy = np.linspace(0.0, 0.07, 8) + 0.41
#positionsz = np.linspace(0.0, 0.4, 6) + 0.04

#positionsy = np.linspace(0.0, 0.1, 9) + 0.44
#positionsz = np.linspace(0.0, 0.1, 9) + 0.44


Fy = []
Fz = []
ry = []
rz = []

fig = plt.figure()

for posy in positionsy:
	for posz in positionsz:
		if posy < 0.7 and posz < 0.7:
			strposy = str(posy)
			strposz = str(posz)
			
			if strposy == "0.0":
				strposy = "0"
			
			full_folder = prefix + case + strposy + "x" + strposz
					
			files = sorted(glob.glob(full_folder + "/pinning_force/*.txt"))
			
			print files
			
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
			
#			if posy != posz:
#				Fy.append(mz)
#				Fz.append(my)
#				
#				ry.append(posz)
#				rz.append(posy)
	
		
#print Fy
#print Fz

dump_plots(rz, ry, Fz, -np.array(Fy))

#plottraj("case_0.05_20.0__80_40_1.5__20x70")
#plottraj("case_0.05_20.0__80_40_1.5__16x80")
#plottraj("case_0.05_20.0__80_40_1.5__32.5x63")
#plottraj("case_0.05_20.0__80_40_1.5__30x59")

plt.show()
fig.savefig("/home/alexeedm/udevicex/media/square_duct_raw.pdf", bbox_inches='tight', transparent=True)


