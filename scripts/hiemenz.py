#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math
import scipy
from scipy import interpolate

import matplotlib.pyplot as plt
	
def hiemenz_profile(fname, hbin, maxDistU, maxDistV, stagPoint2, h2):	
	
	nbinsU = int(maxDistU / hbin)
	nbinsV = int(maxDistV / hbin)
	
	u = np.zeros(nbinsU)
	v = np.zeros(nbinsV)
	
	countU = np.zeros(nbinsU)
	countV = np.zeros(nbinsV)

	f = h5py.File(fname, 'r')
	m = f["momentum"][()]
	d = f["density"][()]
		
	m = np.squeeze(m.mean(axis=1))
	d = np.squeeze(d.mean(axis=1))
		
	it = np.nditer(m, flags=['multi_index'])
	r = np.zeros(2)
	
	while not it.finished:
		r[0] = h2[0] * (it.multi_index[0] + 0.5)
		r[1] = h2[1] * (it.multi_index[1] + 0.5)
		
		# plane is (x, z), orth to y
		# my flow is along x, so now x -> y
		# x and z are swapped in h5 (wtf)
		x = r[0] - stagPoint2[0]
		y = stagPoint2[1] - r[1]

		
		if abs(x) < 5 and y < maxDistU and y > 0:
			
			thisU = m[it.multi_index[0], it.multi_index[1], 2]
						
			ibin = int(y*nbinsU / maxDistU)
			countU[ibin] += 1
			
			u[ibin] += thisU / x

		if abs(x) < 5 and y < maxDistV and y > 0:
			
			thisV = m[it.multi_index[0], it.multi_index[1], 0]
						
			ibin = int(y*nbinsV / maxDistV)
			countV[ibin] += 1
			
			v[ibin] += thisV
			
		
		it.iternext()
		it.iternext()
		it.iternext()
	
	u /= countU
	v /= countV
	
	#print u
	#print v
			
	return u, v

def derivatives(y, t):
	return [y[1], y[2],  -y[0]*y[2] + y[1]*y[1] - 1]

def ic(a):
	return [0, 0, a]

def solve_const_a(a):
	t = np.linspace(0, 10, 100)
	y0 = ic(a)
	
	y = scipy.integrate.odeint(derivatives, y0, t)
	
	return y[-1, 1] - 1

def F_Fprime(t):
	bc = scipy.optimize.root(solve_const_a, 1.23)
	
	print "Boundary value: ", bc.x
	y0 = ic(bc.x)

	return scipy.integrate.odeint(derivatives, y0, t)
	

# half profile
def fit_uv(u, v, h):
	
	xU = np.linspace(h/2, u.size*h + h/2, u.size)
	xV = np.linspace(h/2, v.size*h + h/2, v.size)
	
	x_maxU = np.max(xU)
	x_maxV = np.max(xV)

	fu = scipy.interpolate.interp1d(xU, u)
	fv = scipy.interpolate.interp1d(xV, v)
	
	t = np.linspace(0, 5, 100)
	theory = F_Fprime(t)
	
	nu=39/8.0
	
	def l2norm(x):
		k  = x[0]
		eta = math.sqrt(abs(k / nu))
		
		tmp = t/eta
		
		ref_t = tmp[ np.where(tmp < min(x_maxU, x_maxV)) ]
		top = ref_t.size
		ref_t = ref_t[ np.where(ref_t > h/2) ]
		l = ref_t.size
		
		ref_u = k                    * theory[top-l:top, 1]
		ref_v = math.sqrt(abs(k*nu)) * theory[top-l:top, 0]
		
		diffV = ref_v - fv(ref_t)
		diffU = ref_u - fu(ref_t)
		l2 = 5*math.sqrt( np.sum(diffU*diffU) ) + math.sqrt( np.sum(diffV*diffV) )
		
		return l2
	
	res = scipy.optimize.minimize(l2norm, [0.1])

	print res.x
	
	k = res.x[0]
	eta = math.sqrt(k / nu)

	return k, nu, eta
	

def dump_plots(u, v, h,   k, nu, eta):
	#fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols,4*nrows), facecolor='white')
	
	u = u / k
	v = v /  math.sqrt(k*nu)

	xU = eta*np.linspace(h/2, u.size*h + h/2, u.size)
	xV = eta*np.linspace(h/2, v.size*h + h/2, v.size)

	#plt.subplot(nrows, ncols, ifig)
	
	t = np.linspace(0, 5, 1000)
	theory = F_Fprime(t)
	
	plt.plot(t, theory[:, 1], label="U analytical", color="C0")
	plt.plot(t, theory[:, 0], label="V analytical", color="C1")
	
	plt.plot(xU, u, 'o', label="U, dpd", mfc='none', mew=2, color="C0")
	plt.plot(xV, v, 'o', label="V, dpd", mfc='none', mew=2, color="C1")
	
	plt.xlabel(r'$\eta$', fontsize=16)
	plt.ylabel('velocity', fontsize=16)
	
	
	ax=plt.gca()
	ax.set_ylim([0, 1.2])
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	ax.yaxis.major.formatter._useMathText = True
	ax.set_xlim([0, 2.5])
	plt.grid()
	plt.legend(fontsize=14)

	plt.subplots_adjust(wspace=0.3)
	plt.subplots_adjust(hspace=0.3)

	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

def main():
	maxDistU = 5.0
	maxDistV = 8.0
	h = 0.5
	
	#fname = sys.argv[1]
	fname = "/home/alexeedm/extern/daint/scratch/hiemenz/2drun_0.05/xdmf/all.h5"
	
	u, v = hiemenz_profile(fname, h, maxDistU, maxDistV, [32, 52], [0.5, 0.5])
	
	k, nu, eta = fit_uv(u, v, h)
	
	print k, "  ", nu, "  ", eta
		
	dump_plots(u, v, h,   k, nu, eta)


if __name__ == "__main__":
	main()
