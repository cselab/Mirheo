#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
	
def average_files(fnames):
	init = 0
	
	for fname in fnames:
		print fname

		f = h5py.File(fname, 'r')
		m = f["momentum"]
		d = f["density"]
			
		if init == 0:
			momentum = m[()]
			density  = d[()]
			init = 1
		else:
			momentum[:] += m[:]
			density[:]  += d[:]

	avgU = momentum[()]
	avgD = density[()]

	avgU = np.squeeze(avgU.mean(axis=2))
	avgU = np.squeeze(avgU.mean(axis=0))
	
	avgD = np.squeeze(avgD.mean(axis=2))
	avgD = np.squeeze(avgD.mean(axis=0))
	
	avgU = np.squeeze(avgU[:, 0])
	
	avgU = np.fabs(avgU)
	
	return avgD, avgU


# half profile
def fit_velocity(profile, gz, rho):
	x = np.linspace(0.25, profile.size*0.5 - 0.25, profile.size)
	
	[coeff, residuals, rank, sv, rcond] = np.polyfit(x, profile, 2, full=True)
	p = np.poly1d(coeff)
	e_norm = math.sqrt(residuals)/max(profile)*100. # normalized error

	D = profile.size*0.5
	avgvel = np.mean(profile)

	eta = rho * gz * D*D / (12.0 * avgvel)

	return p, eta, e_norm

def dump_plots(density, velocity, velFit):
	ifig = 0
	nrows = 1
	ncols = 2
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols,4*nrows), facecolor='white')

	x = np.linspace(0.25, density.size*0.5 - 0.25, density.size)


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
	
    #figpath = "%s/profiles.png" % (resdir)
    #plt.savefig(figpath, bbox_inches='tight')
    #plt.close(fig)

def main():
	dens, vel = average_files(sys.argv[1:])
	
	dens = dens[0: dens.size/2]
	vel = vel[0: vel.size/2]
	
	velFit, eta, err = fit_velocity(vel, 0.05, 8)
	
	print "Viscosity: ", eta
	print "Fit err: ", err
	
	dump_plots(dens, vel, velFit)


if __name__ == "__main__":
	main()
