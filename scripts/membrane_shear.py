#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:44:20 2018

@author: alexeedm
"""

import re
import numpy as np
import glob
import matplotlib.pyplot as plt
import math


def dump_plots(Dxy, theta, deltaT, G, ttf):
	
	ifig = 0
	nrows = 1
	ncols = 2
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols,4*nrows), facecolor='white')

	x = np.linspace(0, len(Dxy)*deltaT, len(Dxy))
	
	ifig = ifig+1
	plt.subplot(nrows, ncols, ifig)
	plt.plot(x, Dxy)
	plt.xlabel('t', fontsize=14)
	plt.ylabel('Dxy', fontsize=14)
	plt.grid()

	ifig = ifig+1
	plt.subplot(nrows, ncols, ifig)
	plt.plot(x, theta)
	plt.xlabel('t', fontsize=14)
	plt.ylabel('Theta', fontsize=14)
	plt.ylim([0.0, 0.25])
	plt.grid()
	
	plt.suptitle("G = " + ("%.2f" % G) + ", frequency = " + ("%.3f" % ttf), fontsize=16)
	plt.legend(fontsize=14)

	plt.subplots_adjust(wspace=0.3)
	plt.subplots_adjust(hspace=0.3)
	
	return fig


#%%

folder = "/home/alexeedm/extern/daint/scratch/membrane_shear/case_0.5_5.0/"
#folder = "/home/alexeedm/udevicex/apps/udevicex" 
script = folder + "/script.xml"

vels = []
for line in open(script).readlines():
	if 'moving_plane' in line:
		m = re.search(r'velocity\s*=\s*"\s*(\S+)', line)
		vels.append(float(m.group(1)))
		
vels = sorted(vels)

L = 28.0
mu = 38.0
a = 5.0
Y = 13.57142857
k = (vels[1] - vels[0]) / L

dt = 1e-3
dump_freq = 2000

G = a * k * mu / Y

trajectories = None

xyzfiles = sorted(glob.glob(folder + "/xyz/capsule*"))
for f in xyzfiles:
	lines = open(f).readlines()
	
	coords = np.zeros((0, 3))
	
	for l in lines[2:]:
		coords = np.vstack( (coords, np.array([float(x) for x in l.split()[1:]])) )
	
	if trajectories == None:
		trajectories = coords
	else:
		trajectories = np.dstack((trajectories, coords))
	
trajectories = np.swapaxes(trajectories, 1, 2)


#%% Elongation and inclination and TTF

theta = []
Dxy = []

proj_long = np.zeros(trajectories.shape[0:2])

for t in range(0, trajectories.shape[1]):
	
	com = np.mean(np.squeeze(trajectories[:, t, :]), axis = 0)
		
	I = np.zeros((3, 3))
	E = np.identity(3)
	
	rs = trajectories[:, t, :] - com
	
	I += np.sum( np.matmul(rs[:, np.newaxis, :], rs[:, :, np.newaxis]), axis = 0 ) * E
	I -= np.sum( np.matmul(rs[:, :, np.newaxis], rs[:, np.newaxis, :]), axis = 0 )

	lm, axes = np.linalg.eig(I)
	axes *= axes[0] / np.abs(axes[0]) # make consistent sign
	
	newcoos = np.matmul( rs[:, np.newaxis, :], axes )
	lo = np.squeeze( np.min(newcoos, axis=0) )
	hi = np.squeeze( np.max(newcoos, axis=0) )
	
	L = np.max(hi - lo)
	l = np.min(hi - lo)
		
	longid = np.argmax(hi - lo)
	
	Dxy.append((L-l)/(L+l))
	theta.append( np.math.acos(abs(axes[0, longid]) ) / math.pi )
	
	proj_long[:, t] = np.matmul(trajectories[:, t, :] - com, axes[:, longid])


#%% TTF

mult = 20
omega = []
amp = []
for n in range(0, proj_long.shape[0]):
	
	T = len(proj_long[n])
	raw = proj_long[n, T/4:T]
	
	signal = np.pad(raw, mult*len(raw), 'constant' )
	
	freq = np.abs(np.fft.fft(signal))
	
#	for ind in range(0, len(freq)-1):
#		if freq[ind] > freq[ind+1]:
#			freq[ind] = 0
#		else:
#			break
	
#	plt.plot(raw - raw[0])
	
	omega.append( np.argmax(freq[0:len(freq)/2]) )

omega = np.array(omega)
omega_filt = omega[omega > np.max(omega) * 0.75]

lowest_freq = 1.0 / (dt * dump_freq * len(signal))
ttf = 4.0*math.pi / k * (np.average(omega_filt) * lowest_freq)

print ttf 

#plt.plot(omega_filt)
#plt.show()

fig = dump_plots(Dxy, theta, dt*dump_freq * k, G, ttf)
#fig.savefig("./image.png")

plt.show()























