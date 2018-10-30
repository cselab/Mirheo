#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs = "+")
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

    nodes = data[:,0]
    times = data[:,1]

    nref = nodes[args.ref]
    tref = times[args.ref]

    tref = tref * nref

    speedup = tref / times
    efficiency = speedup / nodes

    ax.plot(nodes, speedup, "-D", label=getDomain(file))    
        
nodes = [8, 27, 64, 125, 216]
ax.set_yticks(nodes)
ax.set_yticklabels(np.array(nodes, dtype=int))

ax.set_xticks(nodes)
ax.set_xticklabels(np.array(nodes, dtype=int))


ax.plot(nodes, nodes, "k--", label="ideal")

ax.set_xlabel("nodes")
ax.set_ylabel("speedup")
plt.legend(frameon=False)
plt.tight_layout()

#ax.grid()

#print(nodes, efficiency)

if args.out == "gui":
    plt.show()
else:
    plt.savefig(args.out)
