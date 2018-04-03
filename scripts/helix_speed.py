#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:28:41 2018

@author: alexeedm
"""
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl


r = 5
L = 36
nu = 100
rho = 8
basefolder = "/home/alexeedm/extern/daint/scratch/helix/"

cases = glob.glob(basefolder + "case_*")

deps = dict()
for case in cases:
	f = case + "/pos/helix.txt"
	lines = open(f).readlines()
	
	omega = np.mean(np.array([ float(x.split()[14]) for x in lines ]))
	vel = ( float(lines[-1].split()[4]) - float(lines[0].split()[4]) ) / float(lines[-1].split()[1])
	
	(R, torque) = re.match(".*case_(\d+)_(\d+).*", case).groups()
	R = float(R)
	
	if R in deps:
		deps[R].append([float(torque), omega, vel])
	else:
		deps[R] = [[float(torque), omega, vel]]


fig,ax = plt.subplots(1)
#ax.set_yscale("log", nonposy='clip')
ax.set_ylim([0.18, 0.25])
#ax.add_patch(mpl.patches.Rectangle((0,0.38197), 300000, 1.145916 - 0.38197, facecolor="black", alpha=0.2))

for R in sorted(deps):
	
	data = np.array(deps[R])
	data = data[data[:,0].argsort()]
	
	x = data[:,0]
	y =  (data[:,2]*L) / (data[:,1]*r*r) 
	
	plt.plot(x, y, "-o", ms=5, label="R = " + str(2*R))



 
plt.xlabel('Torque', fontsize=16)
plt.ylabel(r'$\frac{Re_{\omega}}{Re_{t}}$', fontsize=20)
plt.grid()
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()