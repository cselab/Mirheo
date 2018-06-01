#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:59:10 2018

@author: alexeedm
"""

def opendraw(fname, label):
	raw = open(fname).readlines()

	x    = np.array( [float(l.split()[0]) for l in raw][1:] )
	data = np.array( [float(l.split()[1]) for l in raw][1:] )
	
	data = data[0] / data
	
	plt.plot(x, data,  '-o', label=label,  ms="6")
	
	return x

import matplotlib.pyplot as plt
import numpy as np

fname192 = "/home/alexeedm/extern/daint/scratch/strong_scale/raw192.txt"
fname288 = "/home/alexeedm/extern/daint/scratch/strong_scale/raw288.txt"
fname384 = "/home/alexeedm/extern/daint/scratch/strong_scale/raw384.txt"

opendraw(fname192, r'Domain $192^3$')
opendraw(fname288, r'Domain $288^3$')
x = opendraw(fname384, r'Domain $384^3$')

ideal = x / x[0]
plt.plot(x, ideal, '--', label="Ideal")

plt.xlabel('Nodes', fontsize=16)
plt.ylabel('Speedup', fontsize=16)

plt.legend(fontsize=14)
#plt.grid()
plt.grid(axis='both')

plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.3)

plt.xticks(x)
plt.yticks(x / x[0])

plt.show()

figpath = "/home/alexeedm/udevicex/media/strong.pdf"
plt.savefig(figpath, bbox_inches='tight')