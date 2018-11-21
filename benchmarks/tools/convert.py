#! /usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--icfile', type=str, required = True)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

data = np.loadtxt(args.icfile)
n = len(data)

matrices = np.zeros((4*n, 4))

for i in range(n):
    rq = data[i,:]
    matrices[4*i + 0,:] = [1.0, 0.0, 0.0, rq[0]]
    matrices[4*i + 1,:] = [0.0, 1.0, 0.0, rq[1]]
    matrices[4*i + 2,:] = [0.0, 0.0, 1.0, rq[2]]
    matrices[4*i + 3,:] = [0.0, 0.0, 0.0, 1.0]

np.savetxt(args.out, matrices)
