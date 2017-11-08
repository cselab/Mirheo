#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
def radial_profile(fname, nbins, h2, cen, radius):
	vel  = np.zeros(nbins+1)
	dens = np.zeros(nbins+1)
	cnt  = np.zeros(nbins+1)
	
	f = h5py.File(fname, 'r')
	m = f["momentum"]
	d = f["density"]
		
	m = np.squeeze(np.mean(m, axis=2))
	
	d = np.squeeze(np.mean(d, axis=2))
	
	a = np.squeeze(np.mean(m, axis=0))
	a = np.squeeze(np.mean(a, axis=0))	
	
	it = np.nditer(m, flags=['multi_index'])
	x = np.zeros(2)
	
	while not it.finished:
		x[0] = h2[0] * (it.multi_index[0] + 0.5)
		x[1] = h2[1] * (it.multi_index[1] + 0.5)
		
		r = math.sqrt( (x[0] - cen[0])**2 + (x[1] - cen[0])**2 )
				
		ibin = int(r*nbins / radius)
		
		if ibin <= nbins:
			mydens = d[it.multi_index[0], it.multi_index[1]]
			vel[ibin]  += it[0]  # x component
			dens[ibin] += mydens
			cnt[ibin]  += 1
		
		it.iternext()
		it.iternext()
		it.iternext()
	
	vel  /= cnt
	dens /= cnt
			
	return vel, dens, cnt


# half profile
def fit_velocity(profile, weights, gz, rho, h):
	x = np.linspace(h/2, profile.size*h - h/2, profile.size)
	
	[coeff, residuals, rank, sv, rcond] = np.polyfit(x, profile, 2, full=True)
	p = np.poly1d(coeff)
	e_norm = math.sqrt(residuals)/max(profile)*100.0 # normalized error

	r = profile.size*h
	
	avgvel = np.sum(profile * weights) / np.sum(weights)

	eta = rho * gz * r*r / (8.0 * avgvel)

	return p, eta, e_norm

def dump_plots(velocity, velFit, density, h):
	ifig = 0
	nrows = 1
	ncols = 2
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols,4*nrows), facecolor='white')

	x = np.linspace(h/2, velocity.size*h - h/2, velocity.size)

	ifig = ifig+1
	plt.subplot(nrows, ncols, ifig)
	plt.plot(x, density)
	plt.xlabel('x')
	plt.ylabel('density')
	ax=plt.gca()
	ax.set_ylim([0, max(density)+2])
	ax.set_xlim([0, max(x)])
	plt.xticks(np.arange(0, max(x)+1, 5.0))
	plt.grid()

	ifig = ifig+1
	plt.subplot(nrows, ncols, ifig)
	plt.plot(x, velocity, label="velocity")
	plt.plot(x, velFit(x), label="fit")
	
	plt.xlabel('x')
	plt.ylabel('velocity')

	ax=plt.gca()
	ax.set_ylim([0, max(velocity)*1.2])
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	ax.yaxis.major.formatter._useMathText = True
	ax.set_xlim([0, max(x)])
	plt.xticks(np.arange(0, max(x)+1, 5.0))
	plt.legend()
	plt.grid()

	plt.subplots_adjust(wspace=0.3)
	plt.subplots_adjust(hspace=0.3)

	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

def main():
	nbins = 100
	r = 30.0
	vel, dens, cnt = radial_profile(sys.argv[1], nbins, [0.125, 0.125], [32, 32], r)
	
	velFit, eta, err = fit_velocity(vel, cnt, float(sys.argv[2]), 8, r / nbins)
	
	print "Viscosity: ", eta
	print "Fit err: ", err
	
	dump_plots(vel, velFit, dens, r / nbins)


if __name__ == "__main__":
	main()
