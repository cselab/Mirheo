#! /usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--icfile', type=str, required = True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--format', type=str, required=True,
                    help = "'Diego' or 'Sergey'",
                    choices=['Diego', 'Sergey'])

args = parser.parse_args()

data = np.loadtxt(args.icfile)
n = len(data)

if args.format == 'Sergey':
    matrices = np.zeros((4*n, 4))

    for i in range(n):
        rq = data[i,:]
        matrices[4*i + 0,:] = [1.0, 0.0, 0.0, rq[0]]
        matrices[4*i + 1,:] = [0.0, 1.0, 0.0, rq[1]]
        matrices[4*i + 2,:] = [0.0, 0.0, 1.0, rq[2]]
        matrices[4*i + 3,:] = [0.0, 0.0, 0.0, 1.0]

elif args.format == 'Diego':
    matrices = np.zeros((n, 3+16))

    for i in range(n):
        rq = data[i,:]
        matrices[i,:] = [rq[0], rq[1], rq[2],
                         1.0, 0.0, 0.0, rq[0],
                         0.0, 1.0, 0.0, rq[1],
                         0.0, 0.0, 1.0, rq[2],
                         0.0, 0.0, 0.0, 1.0]


np.savetxt(args.out, matrices)
