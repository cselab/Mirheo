#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (8, 16, 4)

u = udx.udevicex(args.ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=3)
u.registerParticleVector(pv=pv, ic=ic)

sampleEvery = 1
dumpEvery   = 1
binSize     = (1., 1., 1.)

field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(3)

# TEST: dump.h5
# cd dump
# rm -rf h5
# udx.run --runargs "-n 2" ./h5.py --ranks 1 1 1 > /dev/null
# udx.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt
