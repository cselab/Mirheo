#!/usr/bin/env python

import numpy as np
import scipy as sp
import h5py
import math

import matplotlib.pyplot as plt
import glob
	
def radial_profile(fname, nbins, h2, cen, radius):
	vel  = np.zeros(nbins)
	dens = np.zeros(nbins)
	cnt  = np.zeros(nbins)
	
	f = h5py.File(fname, 'r')
	
	m = f["velocity"]
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
		
		if ibin < nbins:
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

	eta = rho * gz * r*r / (4.0 * avgvel)

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
	plt.plot(x, velFit(x), label="Analytical", color="C0")
	plt.plot(x[::3], velocity[::3], 'o', label="Simulation", color="C0", ms="5", mfc='none')
	
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

nbins = 40
r = 20.0

folder = "/home/alexeedm/extern/daint/scratch/poiseuille/"
case = "case_160_0.9*"
xdmfs = sorted(glob.glob(folder + case + "/xdmf/*.h5"))

mus = []
errs = []
finalVel = np.array([])
finalFit = np.array([])

for fname in xdmfs:
	print fname
	vel, dens, cnt = radial_profile(fname, nbins, [0.25, 0.25], [22, 22], r)
	velFit, mu, err = fit_velocity(vel, cnt, 0.1, 8, r / nbins)	
	
	finalVel = vel
	finalFit = velFit
	
	print mu, err
	print ""
	
	mus.append(mu)
	errs.append(err)

#%%

func = lambda t, a,b,t0 : a / (1 - np.exp( np.minimum(-b*(t+0.1*t0), 5)  ))
efunc = lambda t, a,b : a/(t**b)

x = np.linspace(0, 1, len(mus)+1)[1:]

skip = 1
x_ = x[skip:]
mus_ = np.array(mus)[skip:]
errs_ = np.array(errs)[skip:]


params,  cov = sp.optimize.curve_fit( func, x_, mus_ )
eparams, cov = sp.optimize.curve_fit( efunc, x_, errs_)

ax = plt.subplot()
#ax.set_yscale("log")
plt.errorbar(x_, mus_, yerr=0.01*errs_, fmt="o")
plt.plot(x_, errs_, "x", ms=5)
plt.plot(x_, func(x_, params[0], params[1], params[2]))
plt.plot(x_, efunc(x_, eparams[0], eparams[1]))
plt.show()

print params[0], eparams[0]

#	dump_plots(vel, velFit, dens, r / nbins)









