#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:46:52 2017

@author: alexeedm
"""

#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
def couette_profile(fname):
	
	f = h5py.File(fname, 'r')
	m = f["momentum"]
	d = f["density"]
		
	m = np.squeeze(np.mean(m, axis=1))
	m = np.squeeze(np.mean(m, axis=1))
	
	return m[:, 0]

def fit_velocity(profile, h):
	x = np.linspace(h/2, profile.size*h - h/2, profile.size)
	
	[coeff, residuals, rank, sv, rcond] = np.polyfit(x, profile, 1, full=True)
	p = np.poly1d(coeff)
	
	print coeff

	return p


def dump_plots(velocity, velfit, h):
	ifig = 0
	nrows = 1
	ncols = 1
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols,4*nrows), facecolor='white')

	x = np.linspace(h/2, velocity.size*h - h/2, velocity.size)

	ifig = ifig+1
	plt.subplot(nrows, ncols, ifig)
	plt.plot(x, velfit(x), label="Analytical", color="C0")
	plt.plot(x, velocity, 'o', label="Simulation", color="C0", ms="5", mfc='none')
	
	plt.xlabel('r', fontsize=16)
	plt.ylabel('velocity', fontsize=16)

	#ax=plt.gca()
	#ax.set_ylim([0, max(velocity)*1.2])
	#ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	#ax.yaxis.major.formatter._useMathText = True
	#ax.set_xlim([0, max(x)])
	#plt.xticks(np.arange(0, max(x)+1, 5.0))
	plt.legend(fontsize=14)
	plt.grid()

	plt.subplots_adjust(wspace=0.3)
	plt.subplots_adjust(hspace=0.3)

	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

def main():

	fname = "/home/alexeedm/extern/daint/scratch/sphere_lift/run_10_50/xdmf/avg_rho_u00060.h5"
	
	vel = couette_profile(fname)
	vel = vel[4:-4]
	
	velfit = fit_velocity(vel, 0.5)

	dump_plots(vel, velfit, 0.5)


if __name__ == "__main__":
	main()
