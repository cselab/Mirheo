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
		
	return lines[:,2], lines[:,3]


def dump_plots(x, y, r):
	
	t   = np.linspace(0, 2*math.pi, 200)
	xth = np.sin(t)
	yth = np.cos(t)

	fig = plt.figure()
	
	plt.plot(r*xth, r*yth, label="Analytical", color="C0")
	plt.plot(10*xth, 10*yth, color="black", lw=2)
	plt.plot(30*xth, 30*yth, color="black", lw=2)
	plt.plot(x-32, y-32, '.', label="Simulation", alpha=.25)#, mfc='none', mew=2, color="C0")
	
	plt.xlabel('x', fontsize=16)
	plt.ylabel('y', fontsize=16)
	
	fig.tight_layout()
	
	ax=plt.gca()
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	#plt.grid()
	plt.legend(fontsize=14)
	
	plt.axes().set_aspect('equal', 'datalim')

	plt.show()


def main():
	
	fname = "/home/alexeedm/udevicex/apps/udevicex/tc_pos/all.txt"

	
	x, y = trajectory(fname)
	
	x = x[20:]
	y = y[20:]

	dump_plots(x, y, 20.5)


if __name__ == "__main__":
	main()
