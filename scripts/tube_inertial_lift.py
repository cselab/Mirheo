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


def coefficient(frc, rho, u, r, R):
	return frc / (rho * u**2 * (2*r)**4 / (2*R)**2)

def mean_err_cut(vals):
	npvals = np.array(vals[20:]).astype(np.float)
	
	m = np.mean(npvals)
	v = np.var(npvals) / npvals.size
	
	return m,v

def dump_plots(positions, alldata, ref):
	
	plt.plot(ref[:,0], ref[:,1], ":D", ms=8, linewidth=5, label="Chao Liu et al. Lab on a Chip (2015)", zorder=2)
		
	for data, err, label, fmt in alldata:
		plt.errorbar(positions, data, yerr=err, fmt=fmt, ms=7, linewidth=1.5, label=label, zorder=3)

	plt.xlabel('y/R', fontsize=16)
	plt.ylabel('Cl', fontsize=16)
	plt.grid()
	plt.legend(fontsize=11.5)


	plt.tight_layout()
	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

## ratio = 0.166
ref = np.array([
0, 0.0001526135062951961,
0.1, 0.06814193056085455,
0.2, 0.14208317436093088,
0.3, 0.16955360549408616,
0.4, 0.15742083174360924,
0.5, 0.10339565051507052,
0.6, 0.026020602823349753,
0.7, -0.18733307897748996
]).reshape([-1, 2])

## ratio = 0.15
#ref = np.array([0.0004303640088072491, 0.00040587219343701797,
#				0.10036231090273762, 0.06927461139896374,
#				0.20029809177447455, 0.1451554404145082,
#				0.3005244333928515, 0.1724525043177897,
#				0.40010050498964855, 0.17044905008635639,
#				0.5002811127299022, 0.11410189982728863,
#				0.6001194010231242, 0.011675302245250485,
#				0.6999018227825918, -0.19292746113989645	]).reshape([8, 2])

def get_forces(case, Um):
	prefix = ""	
	rho = 8.0
	r = 5
	R = 33.333
	
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

alldata = []

#alldata.append( get_forces("/home/alexeedm/extern/daint/project/alexeedm/focusing_liftparams/case_5_0.1__80_20_1.5__") + ("Rigid", "-.o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/project/alexeedm/focusing_liftparams/case_norot_5_0.1__80_20_1.5__") + ("Rigid, no rotation", "-.o") )

#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_5_0.0502__80_25_1.5__", 4.57) + (r'$\lambda = 0.2$', "-o") )
alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_5_0.05177__110_25_2.0__", 4.644375) + (r'$\lambda = 0.2$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_5_0.0345__80_20_1.5__", 3.73) + (r'$\lambda = 0.2$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_8_0.0311__80_40_1.5__", 4.611298828) + (r'$\lambda = 0.2$', "-o") )


#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_soft/case_noforce_0.1_0.2__80_20_1.5__") + (r'$\lambda = 0.2$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_soft/case_noforce_0.1_1.0__80_20_1.5__") + (r'$\lambda = 1.0$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_soft/case_noforce_0.1_5.0__80_20_1.5__") + (r'$\lambda = 5.0$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_soft/case_noforce_0.1_20.0__80_20_1.5__") + (r'$\lambda = 20.0$', "-o") )
#alldata.append( get_forces("/home/alexeedm/extern/daint/scratch/focusing_soft/case_noforce_0.1_40.0__80_20_1.5__") + (r'$\lambda = 40.0$', "-o") )

print alldata
#print Cls
#print err_Cls

fig = plt.figure()
positions = np.linspace(0.0, 0.7, 8)

dump_plots(positions, alldata, ref)
fig.savefig("/home/alexeedm/udevicex/media/tube_lift_coefficients.pdf", bbox_inches='tight')





