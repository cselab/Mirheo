#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:44:20 2018

@author: alexeedm
"""

import numpy as np
import glob
import math


#%%

folder = "/home/alexeedm/extern/daint/scratch/focusing_soft/case_0.1_5.0__80__1.5__0.7"

dt = 1e-3
dump_freq = 2000


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
	
#%% final process

omega = np.array(omega)
omega_filt = omega[omega > np.max(omega) * 0.75]

lowest_freq = 2.0 * math.pi / (dt * dump_freq * len(signal))
ttf = np.average(omega_filt) * lowest_freq

T = len(theta)
avg_theta = np.average(theta[T/4:T])
avg_Dxy   = np.average(Dxy  [T/4:T])

print avg_theta
print avg_Dxy
print ttf





















