#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:04:28 2018

@author: alexeedm
"""

#!/usr/bin/env python

import sys
import numpy as np
import h5py
import re
import scipy.ndimage as ndimage


init = 0
avg = dict()

fin  = h5py.File(sys.argv[1], 'r')
fout = h5py.File(sys.argv[2], 'w')


datasets = [fin[item] for item in fin.keys()]
smoothed = [fout.create_dataset(ds.name, ds.shape, ds.dtype) for ds in datasets]

for ds, sm in zip(datasets, smoothed):
	sm[:] = ds[:]
	
sm = fout["stress"]

data = sm[:]

data_ind = (data >  0.00001)
none_ind = (data <= 0.00001)
data[data_ind] = 1
data[none_ind] = 0

data   = ndimage.gaussian_filter(data,  sigma=(1, 1, 1, 0), mode="wrap", order=0)
smooth = ndimage.gaussian_filter(sm[:], sigma=(1, 1, 1, 0), mode="wrap", order=0)

smooth[none_ind] = 0
smooth[data_ind] = smooth[data_ind] / data[data_ind]

sm[:] = smooth

fout.close()
