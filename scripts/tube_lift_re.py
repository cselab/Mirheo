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
import math
import re
import pickle


def coefficient(frc, rho, u, r, R):
	return frc / (rho * u**2 * (2*r)**4 / (2*R)**2)

def mean_err_cut(vals):
	npvals = np.array(vals[20:]).astype(np.float)
	
	m = np.mean(npvals)
	v = np.var(npvals) / npvals.size
	
	return m,v

def dump_plots(alldata, Re, kappa):
	
	fig = plt.figure()
	plt.title(r'$Re = ' + str(Re) + r'$, $\kappa = ' + str(kappa) + r'$')

	positions = np.linspace(0.0, 0.7, 8)
	
	for data, err, label, fmt in alldata:
		plt.errorbar(positions, data, yerr=err, fmt=fmt, ms=7, linewidth=1.5, label=label, zorder=3)

	plt.xlabel('y/R', fontsize=16)
	plt.ylabel('Cl', fontsize=16)
	plt.grid()
	plt.legend(fontsize=10, ncol=4)

	plt.tight_layout()
	plt.show()
	fig.savefig("/home/alexeedm/udevicex/media/tube_lift_soft__Re_" + str(Re) + "_kappa_" + str(kappa) + ".pdf", bbox_inches='tight')


def get_forces(case, kappa, f, mu):
	prefix = ""	
	rho = 8.0
	r = 5
	R = r/kappa
	
	Um = 2.0 * R**2 * rho*f / (8*mu)
	
	positions = np.linspace(0.0, 0.7, 8)
	
	Cls = [0]
	err_Cls = [0]
	
	
	for pos in positions:
		if pos < 0.0001:
			continue
		
		strpos = "%.1f" % pos
		full_folder = prefix + case + strpos
		
		files = sorted(glob.glob(full_folder + "/pinning_force/*.txt"))
		lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
			
		fy = [ x.split()[3] for x in lines ]
		
		(my, vy) = mean_err_cut(fy)
		Cls.append(coefficient(my, rho, Um, r, R))
		err_Cls.append(coefficient(3.0*math.sqrt(vy), rho, Um, r, R))
		
	return Cls, err_Cls

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
	
def get_data(folder, Re, kappa, S):
	cases01 = sorted(  glob.glob(folder + "*_0.1/"), key = lambda v : map(float, filter(lambda x : is_number(x), v.split('_'))) )
	
	#print cases01
		
	cases = [ re.match(r'(.*)0.1/', c01).group(1) for c01 in cases01 ]	
	
	fs = get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_5_0.05177__110_25_2.0__", kappa, 0.05177, 24.77)
	alldata = [ fs + (r'$\hat Y = -\infty$, $\lambda = \infty$', '-D')]
	
	for c in cases:
		print c
		
		m = re.search(r'case_(.*?)_(.*?)_(.*?)_.*?__.*?_(.*?)_.*?__', c.split('/')[-1])
		
		f, lbd, Y, gamma = [ float(v) for v in m.groups() ]
		mu = S(gamma)
		
		alldata.append( get_forces(c, kappa, f, mu) + (r'$\hat Y =' + str(Y) + '$, $\lambda = ' + str(lbd) + r'$', '--o') )

	return alldata

s = pickle.load( open('../data/visc_80.0_0.5_backup.pckl', 'rb') )



folder = "/home/alexeedm/extern/daint/scratch/focusing_soft/"
Re = 50
kappa = 0.15

data = get_data(folder + 'case_' + str(Re) + '_' + str(kappa) + '/', Re, kappa, s)
#%%
dump_plots(data, Re, kappa)






