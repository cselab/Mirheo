#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:10:43 2018

@author: alexeedm
"""

import ghalton
import matplotlib.pyplot as plt
import numpy as np

seq = ghalton.Halton(2)
uniform = seq.get(100)

h = 62.5
start = 2
r = 5

length = h - 2*r - 4
center = start + h/2

pts = (np.array(uniform)-0.5) * length + center

plt.scatter(pts[:,0], pts[:,1]);
plt.show

for p in pts:
	print '"',p[0], ' ', p[1],'"'