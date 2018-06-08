#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:52:28 2018

@author: alexeedm
"""

import pickle
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

folder = "/home/alexeedm/extern/daint/scratch/poiseuille/"
cases = sorted(glob.glob(folder + "case_*0.1/"))

visc_map = {}

for case in cases:
	print case
	fname = case + "viscosity.txt"
	
	if os.path.isfile(fname):
		line = open(fname).readline()
		visc = float(line.split()[0])
		err  = float(line.split()[1])
		
		m = re.search(r'case_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)/', case)
		
		a, gamma, kbt, power, dt, rho, f = [ float(m.group(i)) for i in range(1,8) ]
		
		tag = str(a) + "_" + str(power)
		
		if tag in visc_map:
			visc_map[tag].append( (gamma, visc, err) )
		else:
			visc_map[tag] = [(gamma, visc, err)]
		
f = open('../data/visc_backup.pckl', 'wb')
pickle.dump(visc_map, f)
f.close()

#%%

f = open('../data/visc_backup.pckl', 'rb')
vm = pickle.load(f)
f.close()
	
for tag in vm.keys():
	
	lst = vm[tag]
	lst = sorted(lst)
	gammas = np.array( [e[0] for e in lst] )
	viscs  = np.array( [e[1] for e in lst] )	
	errs   = np.array( [e[2] for e in lst] )
	
	s = interpolate.InterpolatedUnivariateSpline(gammas, viscs)
	
	f = open('../data/visc_' + tag + '_backup.pckl', 'wb')
	pickle.dump(s, f)
	f.close()
	
	plt.errorbar(gammas, viscs, yerr=errs, fmt="d", ms=3, label=tag, zorder=10)
	
	g_fine = np.arange(1, 700, 0.5)
	plt.plot( g_fine, s(g_fine), zorder=5 )
	
plt.legend()
plt.show()

