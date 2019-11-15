#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (8, 16, 4)

u = mir.Mirheo(args.ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=3)
u.registerParticleVector(pv, ic)

sample_every = 1
dump_every   = 1
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run(2)

# TEST: dump.h5
# cd dump
# rm -rf h5
# mir.run --runargs "-n 2" ./h5.py --ranks 1 1 1 
# mir.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt

# TEST: dump.h5.mpi
# cd dump
# rm -rf h5
# mir.run --runargs "-n 4" ./h5.py --ranks 1 2 1
# mir.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt
