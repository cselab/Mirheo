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
import scipy.integrate as scintegr
import scipy.interpolate as scinterp

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
	cases = sorted(glob.glob(prefix + "*x*/"))
	
	rho = 8
	r = 5
	kappa = 0.22
	H = 2*r / kappa
	
	alldata = []
	for case in cases:
		print case
		
		forces_raw = read_one( sorted(glob.glob(case + "/pinning_force/*.txt")) )
		m = re.search(r'case_(.*?)_(.*?)_(.*?)_.*?__(.*?)_(.*?)_.*?__(.*?)x(.*)', case.split('/')[-2])		
		f, lbd, Y, a, gamma, ry, rz = [ float(v) for v in m.groups() ]
		
		s = pickle.load( open('../data/visc_' + str(a) + '_0.5_backup.pckl', 'rb') )
		mu = s(gamma)
		
		u = 0.3514425374e-1 * H**2 * rho*f / mu
		
		forces = non_dimentionalize(forces_raw, rho, u, r, H)
		
		# in case of NaNs set the error to something big
		if np.isnan(forces[0]) or np.isnan(forces[1]):
			print "NaNs found in this case"
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
			
			mf = mirror(forces)
			alldata.append([-rz, ry] + [-mf[0], mf[1], -mf[2], mf[3]])			
	
	return np.array(alldata)
	

def fit_gaussian(x, y, err):
	
#	kernel = kr.RBF(length_scale=10.0)
	kernel = kr.Matern(length_scale=0.3, nu=2.0)
	gp = GaussianProcessRegressor(kernel=kernel, alpha=err**2, n_restarts_optimizer=10)
	
	gp.fit(x, y)
	
	return gp
			

def f_mag2(gpY, gpZ, coo):
	
	coo = np.atleast_2d(coo)
	fy = gpY.predict(coo)
	fz = gpZ.predict(coo)
	
	mag = fy*fy + fz*fz
		
	return mag

	
def heteroclinic_orbit(gpY, gpZ, npoints):
	
	points = []
	for phi in np.linspace(0, np.pi/4.0, npoints):
		
		func = lambda r : f_mag2(gpY, gpZ, r * np.array([np.cos(phi), np.sin(phi)]) )
		
		sol = scopt.minimize(func, [0.6])
		
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
		err[i] = np.abs( np.dot( np.array([sy[i], sz[i]]), direction ) )
		
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

def point_in_triangle2(A,B,C,P):
    v0 = C-A
    v1 = B-A
    v2 = P-A
    cross = lambda u,v: u[0]*v[1]-u[1]*v[0]
    u = cross(v2,v0)
    v = cross(v1,v2)
    d = cross(v1,v0)
    if d<0:
        u,v,d = -u,-v,-d
    return u>=0 and v>=0 and (u+v) <= d

def find_equilibria(gpx, gpy, lo, hi, tolerance):
	
	# http://sci.utah.edu/~hbhatia/pubs/2014_TopoInVis_robustCP.pdf
	
	npts = 40
	
	x = np.linspace(lo[0], hi[0], npts)
	y = np.linspace(lo[1], hi[1], npts)
	X,Y = np.meshgrid(x,y)
	grid = np.array([X.flatten(),Y.flatten()]).T
		
	vx = gpx.predict(grid).reshape((npts, npts, -1))
	vy = gpy.predict(grid).reshape((npts, npts, -1))
		
	h = (hi - lo) / (npts-1)
	
	coarse = []
	
	for i in xrange(npts-1):
		for j in xrange(npts-1):
			
			a = np.array( [vx[i, j], vy[i, j]] ).flatten()
			b = np.array( [vx[i+1, j], vy[i+1, j]] ).flatten()
			c = np.array( [vx[i, j+1], vy[i, j+1]] ).flatten()
			d = np.array( [vx[i+1, j+1], vy[i+1, j+1]] ).flatten()
			
			test1 = point_in_triangle2(a, b, c, [0.0,0.0])
			test2 = point_in_triangle2(c, b, d, [0.0,0.0])
			
			if test1:
				coarse.append( lo + h*np.array([j,i]) + h/3.0 )
				
			if test2:
				coarse.append( lo + h*np.array([j,i]) + 2.0*h/3.0 )
	
	
	if h[0] > tolerance or h[1] > tolerance:
		
		fine = []
		for pt in coarse:
			fine += find_equilibria(gpx, gpy, pt-h, pt+h, tolerance)
			
		return fine
	
	else:
		return coarse


def streamline(gpY, gpZ, start):
	
	func = lambda t, x : np.array([ gpY.predict(np.atleast_2d(x)), gpZ.predict(np.atleast_2d(x)) ]).flatten()
	
	sol = scintegr.solve_ivp(func, t_span=[0, 50], y0=start, atol=1e-10, rtol=1e-7)
	
	return sol.t, sol.y

def mindist(targets, pt):
	
	dist = 1000.0;
	for t in targets:
		dist = min(np.linalg.norm(t-pt), dist)
		
	return dist

def separatrix_cloud(gpY, gpZ):
	
	r = 0.005
	
	print "Looking for critical points"
	
	raw = np.array( find_equilibria(gpY, gpZ, np.array([-0.01, -0.01]), np.array([0.7, 0.7]), 1e-4) )
		
	
	# do a correction
	multiple = raw[ (raw[:,0] - raw[:,1] > -0.1) ]

	for i in range(multiple.shape[0]):
		multiple[i,:] = [ np.max(multiple[i,:]), np.min(multiple[i,:]) ]
			
	multiple = multiple[ ((multiple[:,0] > 0.1) | (multiple[:,0] > 0.1)) ]
	multiple = multiple[ multiple[:,0].argsort(), : ]	
	
	equilibria = np.array(multiple[0])
	for i in xrange(1, multiple.shape[0]):
		diff = np.linalg.norm(multiple[i] - multiple[i-1])
		
		if diff > r:
			equilibria = np.vstack((equilibria, multiple[i]))
	
	print "Equilibrium points identified:"
	print equilibria

	points = np.empty((2, 0))
	

	for eq in equilibria:
		print "Shooting from point ", eq
		for phi in np.linspace(0, 2*np.pi, 10):
			print "    phi = ", phi
			start = r*np.array([ np.sin(phi), np.cos(phi) ]) + eq
			t, y = streamline(gpY, gpZ, start)
			points = np.hstack((points, y))

			d = mindist(equilibria, y[:,-1])
			tries=0
			while d > r and tries < 10:
				tries += 1
				print "    restarting from ", y[:,-1]
				t, y = streamline(gpY, gpZ, y[:,-1])
				points = np.hstack((points, y))
				d = mindist(equilibria, y[:,-1])
				
		print ""
			
	
	return np.atleast_2d(equilibria), np.atleast_2d(points)

def separatrix(points, resolution):
	
	phi = np.arctan2(points[1, :], points[0, :])	
	
	sx = scinterp.Rbf(phi, points[0], smooth=200.0)
	sy = scinterp.Rbf(phi, points[1], smooth=200.0)
	
	x = np.linspace(0, np.pi / 4.0, resolution)
	sep = np.vstack( (sx(x), sy(x)) ).T
	
	lengths = np.empty(sep.shape[0])
	lengths[0] = 0.0
	for i in range(1, sep.shape[0]):
		lengths[i] = lengths[i-1] + np.sqrt( np.dot(sep[i-1] - sep[i], sep[i-1] - sep[i]) )
		
	return lengths, sep
	

#%%
	
Re = 50
kappa = 0.22

lbd = 1.0
Ca = 1.0

for Re in [50, 200]:
	for lbd in [1.0, 5.0, 25.0]:
		for Ca in [1.0, 0.1, 0.01]:
			
			folder = '/home/alexeedm/extern/daint/scratch/focusing_square_massive/casenew_' + str(int(Re)) + '_' + str(kappa) + '/'
			case = 'case_*_' + str(lbd) + '_' + str(Ca) + '_0.22__160_*_3.0__'
			
			fname = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__lambda_' + str(lbd) + '_Ca_' + str(Ca)
			
			alldata = read_all_data(folder + case)
			
			#%%
			
			gpY, gpZ = process_data(alldata)
			
			x = np.linspace(0, 0.7, 40)
			y = np.linspace(0, 0.7, 40)
			X,Y = np.meshgrid(x,y)
			grid = np.array([X.flatten(),Y.flatten()]).T
			
			#%%
						
			equilibria, points = separatrix_cloud(gpY, gpZ)
			
			#%%
			
			t, sep = separatrix(points, 106)
						
			
			#%%
			
			f, sigma = f_along_curve(gpY, gpZ, sep)
						
			fig = plt.figure()
			plt.plot(t, f)
			plt.fill_between(t, f - sigma, f + sigma, color='red', alpha=0.5, linewidth=0)
			plt.title(r'$Re = ' + str(Re) + r',\kappa = ' + str(kappa) + r', \lambda = ' + str(lbd) + r', Ca = ' + str(Ca) + r'$')
			
			if Re == 50:
				plt.axes().set_ylim([-0.05, 0.02])
				
			if Re == 200:
				plt.axes().set_ylim([-0.09, 0.02])
				
			fig.savefig('/home/alexeedm/udevicex/media/square/' + 'force__' + fname + '.pdf', bbox_inches='tight')
			#plt.show()
			plt.close(fig)

			#%%
			
			
			fig = plt.figure()
			plot_settings()
			norm, cmap = draw_quiver(alldata)
			draw_gp(grid, gpY, gpZ, norm, cmap)
			
			plt.plot( sep[:, 0], sep[:, 1], color='C2', linewidth=3.0 )
			plt.scatter( equilibria[:,0], equilibria[:,1], color='C3', s=50.0, zorder=5)
			#plt.scatter( points[0,:], points[1,:], color='C3', s=5.0, zorder=5)
			
			plt.title(r'$Re = ' + str(Re) + r',\kappa = ' + str(kappa) + r', \lambda = ' + str(lbd) + r', Ca = ' + str(Ca) + r'$')
			plt.tight_layout()
			fig.savefig('/home/alexeedm/udevicex/media/square/' + fname + '.pdf', bbox_inches='tight')

			
			#%%
			
			normals = curve_normal(sep)
			
			if Re == 50:
				every = 15
			if Re == 200:
				every = 21
				
			coarse_orbit   = sep    [::every]
			coarse_normals = normals[::every]
			
			print coarse_orbit.shape
			
			h = 0.02
			refname = folder+case+'refine.txt'
			refname = re.sub(r'\*', 'XXX', fname)
			f = open(refname, 'w')
			for p in np.linspace(-h, h, 4):
				coo = coarse_orbit + coarse_normals*p
				plt.scatter( coo[:,0], coo[:,1], color='black' )
				
				for c in coo:
					f.write('"%f  %f"\n' % (c[0], c[1]))
			
			f.close()

			fig.savefig('/home/alexeedm/udevicex/media/square/' + 'refine__' + fname + '.pdf', bbox_inches='tight')
			plt.close(fig)
			#plt.show()

#			
#			plt.plot(sep[:,0], sep[:,1])
#			plt.quiver(sep[:,0], sep[:,1], normals[:,0], normals[:,1])
#			plt.show()
			
			

