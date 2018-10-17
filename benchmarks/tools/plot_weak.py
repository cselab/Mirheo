#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs="+")
parser.add_argument('--ref', type=int, default=0)
args = parser.parse_args()

fig = plt.figure(0)
ax = fig.add_subplot(1, 1, 1)


def getDomain(fname):
    tmp = fname.split("_")[-1]
    return tmp.split(".")[0]

nodes=[]

for file in args.files:

    print(file)
    data = np.loadtxt(file)

    nodes = data[args.ref:,0]
    times = data[args.ref:,1]

    tref = times[0]
    
    efficiency = tref / times

    ax.plot(nodes, efficiency, "-D", label=getDomain(file))

    ax.set_xticks(nodes)
    ax.set_xticklabels(np.array(nodes, dtype=int))

ax.plot(nodes, np.ones(len(nodes)), "k--", label="ideal")

ax.set_xlabel("nodes")
ax.set_ylabel("weak scaling efficiency")

plt.legend()

plt.show()
