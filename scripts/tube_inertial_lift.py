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
import h5py


def coefficient(frc, rho, u, r, R):
	return 0.5 * frc / (rho * u**2 * r**4 / R**2)

def mean_err_cut(vals):
	npvals = np.array(vals[20:]).astype(np.float)
	
	m = np.mean(npvals)
	v = np.var(npvals) / npvals.size
	
	return m,v

def dump_plots(positions, Cls, err_Cls, Cls_norot, err_Cls_norot, ref):
	
	plt.plot(ref[:,0], ref[:,1], "--o", ms=5, label="Liu, Chao, et al. Lab on a Chip (2015)")
	plt.errorbar(positions, Cls, yerr=err_Cls, fmt='-o', ms=7, elinewidth=2, label="DPD")
	plt.errorbar(positions, Cls_norot, yerr=err_Cls_norot, fmt='-D', ms=7, elinewidth=2, label="DPD, inhibited rotation")

	plt.xlabel('y/R', fontsize=16)
	plt.ylabel('Cl', fontsize=16)
	plt.grid()
	plt.legend(fontsize=14)


	plt.tight_layout()
	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

## ratio = 0.166
ref = np.array([0.0004303640088072491, 0.00040587219343701797,
				0.10036285861385286, 0.07027633851468046,
				0.20029987183559916, 0.14841105354058753,
				0.300526213453976, 0.17570811744386902,
				0.40010310661744597, 0.17520725388601077,
				0.5002819342965751, 0.11560449050086374,
				0.6001194010231242, 0.011675302245250485,
				0.6999018227825918, -0.19292746113989645]).reshape([8, 2])

## ratio = 0.15
#ref = np.array([0.0004303640088072491, 0.00040587219343701797,
#				0.10036231090273762, 0.06927461139896374,
#				0.20029809177447455, 0.1451554404145082,
#				0.3005244333928515, 0.1724525043177897,
#				0.40010050498964855, 0.17044905008635639,
#				0.5002811127299022, 0.11410189982728863,
#				0.6001194010231242, 0.011675302245250485,
#				0.6999018227825918, -0.19292746113989645	]).reshape([8, 2])

def get_forces(case):
	prefix = "/home/alexeedm/extern/daint/scratch/focusing_liftparams/"	
	rho = 8.0
	r = 5
	R = 30
	
	positions = np.linspace(0.0, 0.7, 8)
	
	Cls = [0]
	err_Cls = [0]
	
	
	for pos in positions:
		if pos < 0.0001:
			continue
		
		strpos = "%.1f" % pos
		full_folder = prefix + case + strpos
		
	#	h5fname = full_folder + "/xdmf/avg_rho_u00200.h5"
	#	f = h5py.File(h5fname, 'r')
	#	mom = f["momentum"]
	#		
	#	Um = np.amax( np.mean(mom, axis=0) )
		Um = 2 * 4.550105428529214
		
		files = sorted(glob.glob(full_folder + "/pinning_force/sphere*"))
		lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
			
		fy = [ x.split()[3] for x in lines ]
		
		(my, vy) = mean_err_cut(fy)
		Cls.append(coefficient(my, rho, Um, r, R))
		err_Cls.append(coefficient(4.0*math.sqrt(vy), rho, Um, r, R))
		
	return Cls, err_Cls


Cls, err_Cls = get_forces("case_5_0.1__80_20_1.5__")
Cls_norot, err_Cls_norot = get_forces("case_norot_5_0.1__80_20_1.5__")
		
#print Cls
#print err_Cls

fig = plt.figure()
dump_plots(positions, Cls, err_Cls, Cls_norot, err_Cls_norot, ref)
fig.savefig("/home/alexeedm/udevicex/media/tube_lift_coefficients.pdf", bbox_inches='tight')





