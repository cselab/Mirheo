#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--ref', type=int, default=0)
args = parser.parse_args()


data = np.loadtxt(args.file)

nodes = data[:,0]
times = data[:,1]

tref = times[args.ref]

efficiency = tref / times

fig = plt.figure(0)
ax = fig.add_subplot(1, 1, 1)

ax.plot(nodes, efficiency, "k-+")
ax.plot(nodes, np.ones(len(nodes)), "k--")
#ax.set_xscale('log')
#ax.set_yscale('log')

ax.set_xticks(nodes)
ax.set_xticklabels(np.array(nodes, dtype=int))

ax.set_xlabel("nodes")
ax.set_ylabel("weak scaling efficiency")

plt.show()
