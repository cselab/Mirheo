#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
def couette_profile(fname, hbin, r, R, h2, center2):
	
	nbins = int( (R-r) / hbin )
	
	vel = np.zeros(nbins)
	count = np.zeros(nbins)

	f = h5py.File(fname, 'r')
	m = f["momentum"][()]
	d = f["density"][()]
		
	m = np.squeeze(m.mean(axis=0))
	d = np.squeeze(d.mean(axis=0))
	
	
	it = np.nditer(m, flags=['multi_index'])
	coo = np.zeros(2)
	
	while not it.finished:
		coo[0] = h2[0] * (it.multi_index[0] + 0.5)
		coo[1] = h2[1] * (it.multi_index[1] + 0.5)
		
		# plane is (x, z), orth to y
		# my flow is along x, so now x -> y
		# x and z are swapped in h5 (wtf)
		x = coo[0] - center2[0]
		y = coo[1] - center2[1]
		
		dist = math.sqrt( x*x + y*y );
		ibin = int( (dist - r)*nbins / (R-r) )

		if 0 <= ibin and ibin < nbins:
									
			count[ibin] += 1
			
			vy = m[it.multi_index[0], it.multi_index[1], 0]
			vx = m[it.multi_index[0], it.multi_index[1], 1]
			
			#print x, " ", y, " ", vx, " ", vy, " ", ibin
			
			vel[ibin] += math.sqrt( vx*vx + vy*vy )
		
		it.iternext()
		it.iternext()
		it.iternext()
	
	print vel / count
	
	return vel / count


def fit_velocity(r, R, omega):
	x = np.linspace(r, R, num=20)
	
	eta = r/R
	return omega / (1.0 - eta*eta) * (x-r*r/x)


def dump_plots(velocity, velFit, r, R):
	
	x   = np.linspace(0.125 / (R-r), 1 + 0.125 / (R-r), velocity.size)
	xth = np.linspace(0, 1, velFit.size)

	fig = plt.figure()
	
	plt.plot(xth, velFit, label="Analytical", color="C0")
	plt.plot(x, velocity, 'o', label="Simulation", mfc='none', mew=2, color="C0")
	
	plt.xlabel(r'$\frac{r - R_{inner}} {R_{outer} - R_{inner}}$', fontsize=16)
	plt.ylabel('velocity', fontsize=16)
	
	fig.tight_layout()
	
	ax=plt.gca()
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	plt.grid()
	plt.legend(fontsize=14)


	plt.show()


def main():
	
#	fname = "/home/alexeedm/extern/daint/scratch/taylor_couette/run_0.1/xdmf/all.h5"
	fname = "/home/alexeedm/udevicex/apps/udevicex/xdmf/avg_rho_u00002.h5"
	r = 10.0
	R = 30.0
	
	vel = couette_profile(fname, 0.2, r, R, [0.1, 0.1], [32, 32])
	velFit = fit_velocity(r, R, 0.009)

	dump_plots(vel, velFit, r, R)


if __name__ == "__main__":
	main()
