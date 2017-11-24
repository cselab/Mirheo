#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
def trajectory(fname):
	
	text_file = open(fname, "r")
	lines = np.loadtxt(fname)
	
	c = lines[:, 5]
	s = lines[:, 7]
	
	phi = 2*np.arctan2(s, c)
	t = lines[:, 1]
	
	phi = phi[::]
	t = t[::]
	
	phi = np.arctan(np.tan(phi))
		
	
	return phi[125:500], t[125:500] - t[125]


def fit_traj(t):
	a = 3.0
	b = 5.0
	G = 4.0 / 32.0 * 1.009
	
	ref = np.arctan( b/a * np.tan(a*b*G * t / (a**2 + b**2)) )
	
	return ref
	

def dump_plots(phi, t, ref, tref):
	
	fig = plt.figure()
	
	plt.plot(t, -phi, 'o', label="Simulation", color="C0", ms="6", mfc='none')
	plt.plot(tref, ref, label="Analytical",  color="C0")
	
	plt.xlabel('time', fontsize=16)
	plt.ylabel('orientation angle', fontsize=16)
	
	fig.tight_layout()
	
	ax=plt.gca()
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	plt.grid()
	#plt.legend(fontsize=14)

	plt.show()


def main():
	
	fname = "/home/alexeedm/udevicex/apps/udevicex/jeffrey_pos/all.txt"

	
	phi, t = trajectory(fname)
	
	tref = np.linspace(t[0], t[-1], t.size*5)
	ref = fit_traj(tref)

	dump_plots(phi, t, ref, tref)


if __name__ == "__main__":
	main()
