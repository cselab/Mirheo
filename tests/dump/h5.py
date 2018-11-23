#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (8, 16, 4)

u = ymr.ymero(args.ranks, domain, debug_level=8, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=3)
u.registerParticleVector(pv=pv, ic=ic)

sampleEvery = 1
dumpEvery   = 1
binSize     = (1., 1., 1.)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(3)

# TEST: dump.h5
# cd dump
# rm -rf h5
# ymr.run --runargs "-n 2" ./h5.py --ranks 1 1 1 > /dev/null
# ymr.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt

# TEST: dump.h5.mpi
# cd dump
# rm -rf h5
# ymr.run --runargs "-n 4" ./h5.py --ranks 1 2 1 > /dev/null
# ymr.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt
