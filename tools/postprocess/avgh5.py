#! /usr/bin/env python

import sys
import numpy as np
import h5py as h5

def err(s): sys.stderr.write(s)
def shift(a): return a.pop(0)

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

argv = sys.argv

if len(argv) < 4:
    err("usage: %s <[xyz]> <field> <file0.h5> <file1.h5> ... \n"
        "\t xyz   : reduced directions\n"
        "\t field : field name e.g 'velocity'\n"% argv[0])
    exit(1)

shift(argv)
code = shift(argv)
key  = shift(argv)
rdir = decode(code)
all_fname = argv

val_avg = []
first = True

for sample in all_fname:
    f = fopen(sample)
    field = f[key]
    (nz, ny, nx, dim) = field.shape
    nn = [nz, ny, nx]
    fact = 1.0
    for i in rdir: fact = fact / nn[i]
    val = field.value
    val_reduced = np.sum(val, tuple(rdir)) * fact

    if (first):
        val_avg = val_reduced
        first = False
    else:
        val_avg += val_reduced
    f.close()

val_avg = val_avg / len(all_fname)
val_avg = val_avg.reshape(val_avg.shape[0], -1)

if (sys.version_info > (3, 0)):
    np.savetxt(sys.stdout.buffer, val_avg, "%g")
else:
    np.savetxt(sys.stdout, val_avg, "%g")
