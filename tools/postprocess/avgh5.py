#! /usr/bin/env python

import argparse, sys
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

def decode(code):
    rdir = []
    for c in code:
        if   "xX".find(c) >= 0: rdir.append(2)
        elif "yY".find(c) >= 0: rdir.append(1)
        elif "zZ".find(c) >= 0: rdir.append(0)
        else: err("bad code %c, must be in [xXyYzZ]\n" % c); exit(1)
        
    return rdir



parser = argparse.ArgumentParser(description='Compute the average of grid data along given direction(s).')
parser.add_argument('directions',       type=str, help='a string containing the directions to reduce: [xyz], e.g. xy to retain only the z direction')
parser.add_argument('field',            type=str, help='field name to reduce, e.g. "velocities"')
parser.add_argument('files', nargs='+', type=str, help='input h5 files')
parser.add_argument('--verbose', action='store_true', default=False, help='add progress status if enabled')
args = parser.parse_args()

argv = sys.argv

directions = args.directions
key        = args.field
rdir       = decode(directions)
all_fname  = args.files

val_avg = []
first = True

for sample_id, sample in enumerate(all_fname):
    f = fopen(sample)
    field = f[key]
    (nz, ny, nx, dim) = field.shape
    nn = [nz, ny, nx]
    fact = 1.0
    for i in rdir: fact = fact / nn[i]
    val = field[()]
    val_reduced = np.sum(val, tuple(rdir)) * fact

    if (first):
        val_avg = val_reduced
        first = False
    else:
        val_avg += val_reduced
    f.close()

    if args.verbose:
        sys.stderr.write("{} out of {} done.{}".format(sample_id+1, len(all_fname), '\r' if sample_id < len(all_fname)-1 else '\n'))

val_avg = val_avg / len(all_fname)
val_avg = val_avg.reshape(val_avg.shape[0], -1)

if (sys.version_info > (3, 0)):
    np.savetxt(sys.stdout.buffer, val_avg, "%g")
else:
    np.savetxt(sys.stdout, val_avg, "%g")
