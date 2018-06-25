import pickle
from scipy import interpolate
from scipy.optimize import fsolve
import numpy as np

def params(Re, kappa, s):
	
	rho = 8.0
	r = 5.0
	H = 2*r / kappa
	
#	c = np.sqrt(0.1 * 80 * 8)
#	M = 0.206
#	uavg = c*M/(2*kappa)
	
	uavg = 3.5
	
	mu = H*uavg*rho / Re
	#print mu
	
	gamma = fsolve(lambda g : s(g)-mu, mu)[0]	
	f = 28.45415377*Re * mu**2 / (rho**2 * H**3)
	
	return gamma, f

#s = pickle.load(open('../data/visc_80.0_0.5_backup.pckl', 'rb'))
s = pickle.load(open('../data/visc_160.0_0.5_backup.pckl', 'rb'))


for kappa in [0.22]:
	for Re in [50.0, 200.0]:
		gamma, f = params(Re, kappa, s)
		
		if gamma > 35:
			dt = 0.0001
		elif gamma > 25:
			dt = 0.00025
		elif gamma > 15:
			dt = 0.0005
		else:
			dt = 0.001
			
		print '"160 %7.4f 3.0  %8.5f  %.5f"' % (gamma, dt, f)
