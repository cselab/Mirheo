#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:10:43 2018

@author: alexeedm
"""

import ghalton
import matplotlib.pyplot as plt
import numpy as np


def gen_circle(R, center, r):
	seq = ghalton.Halton(2)
	uniform = np.array(seq.get(200))

	circle = np.empty([0, 2])
	for pt in uniform:
		if np.dot(pt-0.5, pt-0.5) < 0.25:
			circle = np.vstack((circle, 2*pt-1))
	
	pts = circle * (R - r - 2) + center
	pts = pts[0:100]
	
	fig, ax = plt.subplots()
	ax.add_artist(plt.Circle([center, center], R-r, ec='C1', fc='none'))
	plt.scatter(pts[:,0], pts[:,1])
	plt.show

	return pts


def gen_square(left, right, bot, top, r):
	seq = ghalton.Halton(2)
	uniform = np.array(seq.get(100))

	lo = np.array([left, bot]) + (r+2)
	hi = np.array([right, top]) - (r+2)

	pts = lo + uniform * (hi-lo)
	
	plt.scatter(pts[:,0], pts[:,1])
	plt.show

	return pts


pts = gen_square(2, 96.59/2, 2, 96.59/2, 5)


for p in pts:
	print '"',p[0], ' ', p[1],'"'