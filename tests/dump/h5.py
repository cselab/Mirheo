#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (8, 16, 4)

u = ymr.ymero(args.ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=3)
u.registerParticleVector(pv, ic)

sample_every = 1
dump_every   = 1
bin_size     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run(2)

# TEST: dump.h5
# cd dump
# rm -rf h5
# ymr.run --runargs "-n 2" ./h5.py --ranks 1 1 1 
# ymr.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt

# TEST: dump.h5.mpi
# cd dump
# rm -rf h5
# ymr.run --runargs "-n 4" ./h5.py --ranks 1 2 1
# ymr.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt
