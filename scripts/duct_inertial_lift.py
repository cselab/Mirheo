#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pickle
import scipy.optimize as scopt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kr

def coefficient(frc, rho, u, r, H):
	return frc / (rho * u**2 * (2*r)**4 / H**2)

def non_dimentionalize(raw, rho, u, r, H):
	res = []
	for component in raw:
		res.append( coefficient(component, rho, u, r, H) )
		
	return res

def mirror(f):
	return [ f[1], f[0], f[3], f[2] ]
	

def mean_err_cut(vals):
	npvals = np.array(vals[20:]).astype(np.float)
	
	m = np.mean(npvals)
	v = np.var(npvals) / npvals.size
		
	return m,v

def read_one(fnames):
	
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in fnames]))
								
	fy = [ x.split()[3] for x in lines ]
	fz = [ x.split()[4] for x in lines ]
	
	(my, vy) = mean_err_cut(fy)
	(mz, vz) = mean_err_cut(fz)
	
	return my, mz, np.sqrt(vy), np.sqrt(vz)

def read_all_data(prefix):
	cases = sorted(glob.glob(prefix + "*x*"))
	
	rho = 8
	r = 5
	kappa = 0.22
	H = 2*r / kappa
	
	alldata = []
	for case in cases:
		print case
		
		forces_raw = read_one( sorted(glob.glob(case + "/pinning_force/*.txt")) )
		m = re.search(r'case_(.*?)_(.*?)_(.*?)_.*?__(.*?)_(.*?)_.*?__(.*?)x(.*)', case.split('/')[-1])		
		f, lbd, Y, a, gamma, ry, rz = [ float(v) for v in m.groups() ]
		
		s = pickle.load( open('../data/visc_' + str(a) + '_0.5_backup.pckl', 'rb') )
		mu = s(gamma)
		
		u = 0.3514425374e-1 * H**2 * rho*f / mu
		
		forces = non_dimentionalize(forces_raw, rho, u, r, H)
		
		# in case of NaNs set the error to something big
		if np.isnan(forces[0]) or np.isnan(forces[1]):
			forces = [ 0, 0, 10, 10 ]
		
		if ry < 0.7 and rz < 0.7:
			
			if ry != rz:
				alldata.append([ry, rz] + forces)
				alldata.append([rz, ry] + mirror(forces))
			else:
				esym = 0.5*(forces[0] + forces[1])
				fsym = 0.5*(forces[2] + forces[3])
				
				alldata.append( [ry, rz] + [esym, esym, fsym, fsym] )
				
			alldata.append([ry, -rz] + [forces[0], -forces[1], forces[2], -forces[3]])			
	
	return np.array(alldata)
	

def fit_gaussian(coos, force, err):
	
#	kernel = kr.RBF(length_scale=10.0)
	kernel = kr.Matern(length_scale=0.3, nu=2.0)
	gp = GaussianProcessRegressor(kernel=kernel, alpha=err**2, n_restarts_optimizer=10)
	
	gp.fit(coos, force)
	
	return gp
			

def f_mag2(gpY, gpZ, coo):
	
	fy = gpY.predict( np.atleast_2d(coo) )
	fz = gpZ.predict( np.atleast_2d(coo) )
	
	return fy*fy + fz*fz
	
	
def heteroclinic_orbit(gpY, gpZ, npoints):
	
	points = []
	for phi in np.linspace(0, np.pi/4.0, npoints):
		
		func = lambda r : f_mag2(gpY, gpZ, r * np.array([np.cos(phi), np.sin(phi)]) )
		
		sol = scopt.minimize(func, [0.3], bounds=[(0.1,0.7)])
		
		r0 = sol.x[0]
		points.append([r0*np.cos(phi), r0*np.sin(phi)])
	
	return np.array(points)

def f_along_curve(gpY, gpZ, coo):
	
	fy, sy = gpY.predict( np.atleast_2d(coo), return_std=True )
	fz, sz = gpZ.predict( np.atleast_2d(coo), return_std=True )
	
	res = np.empty(fy.shape)
	res[0] = 0.0
	res[-1] = 0.0
	
	err = np.empty(fy.shape)
	err[0] = 0.0
	err[-1] = 0.0
	
	for i in range(1, res.shape[0] - 1):
		direction = 0.5 * ( (coo[i] - coo[i-1]) + (coo[i+1] - coo[i]) )
		direction /= np.linalg.norm(direction)
		
#		print direction, fy[i], fz[i]
#		print
		
		res[i] = np.dot( np.array([fy[i], fz[i]]), direction )
		err[i] = np.abs( np.dot( np.array([fy[i] - sy[i], fz[i]] - sz[i]), direction ) - 
						 np.dot( np.array([fy[i] + sy[i], fz[i]] + sz[i]), direction ) )
		
	return res, err


def process_data(data):
	
	coos = data[:, 0:2]
	
	Fy = data[:, 2]
	Fz = data[:, 3]
	ey = data[:, 4]
	ez = data[:, 5]

	return ( fit_gaussian(coos, Fy, ey), fit_gaussian(coos, Fz, ez) )

def plot_settings():
	plt.xlabel('y', fontsize=16)
	plt.ylabel('z', fontsize=16)

	plt.axes().set_xlim([0.0, 0.72])
	plt.axes().set_ylim([0.0, 0.72])

	plt.axes().set_aspect('equal', 'box', anchor='SW')
	
def draw_quiver(data):
	
	ry = data[:,0].copy()
	rz = data[:,1].copy()
	Fy = data[:,2].copy()
	Fz = data[:,3].copy()

	
	lengths = np.sqrt(Fy*Fy + Fz*Fz)
	Fy = Fy / lengths
	Fz = Fz / lengths
	
	norm = matplotlib.colors.LogNorm()
	norm.autoscale(lengths)
	cm = plt.cm.rainbow
	
	sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	sm.set_array([])
	
	plt.colorbar(sm)
	plt.quiver(ry, rz, Fy, Fz, color=cm(norm(lengths)))
#	plt.quiver(ry, rz, _Fy, _Fz, minlength=0)#, scale=1000, width=0.004)
	
	return norm, cm

	
def draw_gp(grid, gpY, gpZ, norm, cm):
	
	Fy = gpY.predict(grid)
	Fz = gpZ.predict(grid)
		
	ry = grid[:,0]
	rz = grid[:,1]
	
	#print F

	lengths = np.sqrt(Fy*Fy + Fz*Fz)
	Fy = Fy / lengths
	Fz = Fz / lengths
	
	plt.quiver(ry, rz, Fy, Fz, alpha=0.5, color=cm(norm(lengths)))
#	plt.colorbar(sm)
	
def curve_normal(curve):
	# https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
	dx_dt = np.gradient(curve[:, 0])
	dy_dt = np.gradient(curve[:, 1])
	velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
	
	ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
	tangent = np.array([1/ds_dt] * 2).transpose() * velocity
	
	tangent_x = tangent[:, 0]
	tangent_y = tangent[:, 1]

	deriv_tangent_x = np.gradient(tangent_x)
	deriv_tangent_y = np.gradient(tangent_y)
	
	dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
	
	length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
	
	normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
	
	return normal

#%%
folder = "/home/alexeedm/extern/daint/scratch/focusing_square_massive/casenew_50_0.22/"
case = "case_0.15337_1.0_1.0_0.22__160_25.7077_3.0__"

alldata = read_all_data(folder + case)

#%%

gpY, gpZ = process_data(alldata)

print gpY.predict(np.array([[0.3, 0.2], [0.6, 0.5]]))
print gpZ.predict(np.array([[0.3, 0.2], [0.6, 0.5]]))

x = np.linspace(0, 0.7, 40)
y = np.linspace(0, 0.7, 40)
X,Y = np.meshgrid(x,y)
grid = np.array([X.flatten(),Y.flatten()]).T

#%%

orbit = heteroclinic_orbit(gpY, gpZ, 51)

#%%

fig = plt.figure()

plot_settings()
norm, cmap = draw_quiver(alldata)
draw_gp(grid, gpY, gpZ, norm, cmap)


plt.plot(orbit[:,0], orbit[:,1])

plt.tight_layout()
plt.show()

#%%

orbitlen = np.empty(orbit.shape[0])
orbitlen[0] = 0.0
for i in range(1, orbit.shape[0]):
	orbitlen[i] = orbitlen[i-1] + np.sqrt( np.dot(orbit[i-1] - orbit[i], orbit[i-1] - orbit[i]) )

f, sigma = f_along_curve(gpY, gpZ, orbit)

print f

fig = plt.figure()
plt.plot(orbitlen, f)
plt.fill_between(orbitlen, f - sigma, f + sigma, color='red', alpha=0.5, linewidth=0)

plt.show()

#%%

normals = curve_normal(orbit)

coarse_orbit   = orbit  [::5]
coarse_normals = normals[::5]
h = 0.02
for p in np.linspace(-h, h, 4):
	coo = coarse_orbit + coarse_normals*p
#	plt.scatter( coo[:,0], coo[:,1], color='black' )
#
#plt.plot(orbit[:,0], orbit[:,1])
##plt.quiver(orbit[:,0], orbit[:,1], normals[:,0], normals[:,1])
#plt.show()

