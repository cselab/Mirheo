#!/usr/bin/env python

import sys
import numpy as np
import h5py
import math
import scipy
from scipy import interpolate

import matplotlib.pyplot as plt
	
def get_profiles(fnames):	
	
	i = 0
	for fname in fnames:
		print fname

		f = h5py.File(fname, 'r')
		m = f["momentum"][()]
		
		m = np.squeeze(m.mean(axis=2).mean(axis=1))
		
		if i == 0:
			profiles = np.zeros((len(fnames), m.shape[0]))

		profiles[i] = m[:,0]
		i = i+1
			
	return profiles


# half profile
def fit_profiles(n, dt, L):
	
	z = np.linspace(0, L, 100)
	profiles = np.zeros((n, 100))
	
	U0 = 0.5
	nu = 39.0 / 8
	omega = 2*math.pi / 100.0
	kappa = math.sqrt(omega/(nu))

	for i in range(0, n):
		profiles[i] = U0 * np.exp(-kappa*z) * np.cos(omega * (dt*i + 2.5) - kappa*z)

	return profiles
	

def dump_plots(profiles, analytical, dt, L):
	
	i = 0
	for profile, theory in zip(profiles, analytical):
		
		z0 = np.linspace(0.5, L+0.5, profile.size)
		f = scipy.interpolate.interp1d(z0, profile)
		
		z = np.linspace(0, L, theory.size)
		plt.plot(theory, z, color="C"+str(i), label = ("Analytical" if i==0 else "") )
		
		z = np.linspace(0.5, L+0.5, profile.size/2)
		plt.plot(np.flip(f(z), 0), z, 'o', mfc='none', mew=2, color="C"+str(i), label = ("Simulation" if i==0 else "") )
		
		i = i+1
		
	
	plt.xlabel('velocity', fontsize=16)
	plt.ylabel('z', fontsize=16)
	
	
	ax=plt.gca()
	ax.set_ylim([0, L*1.2])
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
	ax.yaxis.major.formatter._useMathText = True
	ax.set_xlim([-0.5, 0.5])
	plt.grid()
	plt.legend(fontsize=14)

	plt.subplots_adjust(wspace=0.3)
	plt.subplots_adjust(hspace=0.3)

	plt.show()
#    figpath = "%s/profiles.png" % (resdir)
#    plt.savefig(figpath, bbox_inches='tight')
#    plt.close(fig)

def main():	
	#fname = sys.argv[1]
	
	dt = 25.0
	L = 64
	
	fnames = ["/home/alexeedm/udevicex/apps/udevicex/osc_xdmf/avg_rho_u00020.h5",
		   "/home/alexeedm/udevicex/apps/udevicex/osc_xdmf/avg_rho_u00025.h5",
		   "/home/alexeedm/udevicex/apps/udevicex/osc_xdmf/avg_rho_u00030.h5",
		   "/home/alexeedm/udevicex/apps/udevicex/osc_xdmf/avg_rho_u00035.h5"]
	
	profs = get_profiles(fnames)
	profs = profs[:, 2:-2]
	
	analytical = fit_profiles(len(fnames), dt, L)
	
	dump_plots(profs, analytical, dt, L)


if __name__ == "__main__":
	main()
