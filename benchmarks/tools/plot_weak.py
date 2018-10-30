#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs="+")
parser.add_argument('--ref', type=int, default=0)
parser.add_argument('--out', type=str, default="gui")
parser.add_argument('--fontSize', type=int, default=12)
args = parser.parse_args()

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : args.fontSize}

matplotlib.rc('font', **font)

fig = plt.figure(0)
ax = fig.add_subplot(1, 1, 1)


def getDomain(fname):
    tmp = fname.split("_")[-1]
    return tmp.split(".")[0]

nodes=[]

for file in args.files:
    data = np.loadtxt(file)

    nodes = data[args.ref:,0]
    times = data[args.ref:,1]

    tref = times[0]
    
    efficiency = tref / times

    ax.plot(nodes, efficiency, "-D", label=getDomain(file))

ax.plot(nodes, np.ones(len(nodes)), "k--", label="ideal")

ax.set_xlabel("nodes")
ax.set_ylabel("weak scaling efficiency")

ax.set_xscale('log')
nodes = [8, 27, 64, 125, 512]
ax.set_xticks(nodes)
ax.set_xticklabels(np.array(nodes, dtype=int))

ax.set_ylim([0.9, 1.05])

plt.legend(frameon=False)
plt.tight_layout()

if args.out == "gui":
    plt.show()
else:
    plt.savefig(args.out)
