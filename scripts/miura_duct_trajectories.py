#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:58:21 2018

@author: alexeedm
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def trajectories(case, ceny, cenz, size):
	files = sorted(glob.glob(case + "/pos/*.txt"))
	lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
		
	try:
		y = np.array([ float(x.split()[3]) for x in lines ])[0:]
		z = np.array([ float(x.split()[4]) for x in lines ])[0:]
	except:
		print "Error reading"
		return [], []
	
	y = (y - ceny) / size
	z = (z - cenz) / size
	
	return y[0:4000], z[0:4000]

def plottraj(traj, signs, alpha):
	
	for t in traj:
		y = signs[0]*t[0]
		z = signs[1]*t[1]

		if len(z) < 1:
			continue
	
		plt.plot(y, z, lw=0.1, zorder=1, color="C0", alpha=alpha)
		plt.plot(y[0], z[0],   "x", ms=2, color="C0", zorder=1, alpha=alpha)
		plt.plot(y[-1], z[-1], "o", ms=4, color="C2", markeredgewidth=0.5, markeredgecolor='black', zorder=10, alpha=alpha)


# Duct
folder = "/home/alexeedm/extern/daint/scratch/focusing_square_free/"
#name = "case_8_0.00769__80_20_1.5__"
name = "scattered_5_0.0315__80_20_1.5__"
#name = "case_5_0.03149__80_20_1.5__"
reference = mpimg.imread("/home/alexeedm/Pictures/miura_fig5a.png")

variants = sorted(glob.glob(folder + name + "*/"))

traj = []
for case in variants:
	print case
	traj.append(trajectories(case, 48.295,48.295, 46.295))


#%%

fig = plt.figure()
plt.imshow(reference, extent=(-1,1, -1,1), zorder=0)

plt.axes().set_xlim([-1.1, 1.1])
plt.axes().set_ylim([-1.1, 1.1])


# Mirror
#for t in traj:
#	t = (-t[0], t[1])

plottraj(traj, (-1, 1),  0.25)
plottraj(traj, (1, -1),  0.25)
plottraj(traj, (-1, -1), 0.25)
plottraj(traj, (1, 1), 1.0)


plt.show()
fig.savefig("/home/alexeedm/udevicex/media/square_duct_trajectories_144.pdf", bbox_inches='tight', transparent=True)


