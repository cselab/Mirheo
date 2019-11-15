#! /usr/bin/env python

import argparse
import numpy as np
import h5py as h5

def err(s): sys.stderr.write(s)

def fopen(fname):
    try:
        f = h5.File(fname, "r")
    except IOError:
        err("u.avgh5: fails to open <%s>\n" % fname)
        sys.exit(2)
    return f

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--out',  type=str, required=True)
args = parser.parse_args()


f = fopen(args.file)
pos    = f["position"]
forces = f["forces"][()]
(n, dim) = pos.shape

pos = pos[()]

pos = pos - (np.sum(pos, axis=0)/n)

r     = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
fmagn = np.sqrt(forces[:,0]**2 + forces[:,1]**2 + forces[:,2]**2)

r     = r.reshape((n,1))
fmagn = fmagn.reshape((n,1))

np.savetxt(args.out, np.hstack((r, fmagn, forces)))
