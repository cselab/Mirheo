#!/usr/bin/env python

import math

# http://pages.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2016.pdf
#
# Data Correlation for Drag Coefficient for Sphere
# Faith A. Morrison
#
def single_sphere(Re)
	Ccreep = 24.0/Re
	Clo = 2.6*(Re/5.0) / (1 + math.pow(Re/5.0, 1.52))
	
	tmp = Re / 2.63e5
	Cint = 0.411 * math.pow(tmp, -7.94) / (1 + math.pow(tmp, -8.00)
	
	Chi = 0.25 * (Re/1e6) / (1 + Re/1e6)
	
	return Ccreep + Clo + Cint + Chi

def compute_cd(vel, frc, l, r)
	return 2*frc * l**3 / (math.pi * u**2 * r**2)


	
