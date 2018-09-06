#!/usr/bin/env python

import udevicex as udx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=int)
args = parser.parse_args()


ranks  = (2, 1, 1)
domain = (4, 2, 2)

u = udx.udevicex(ranks, domain, debug_level=8, log_filename='log')

pos = [[1., 0.25, 0.5],
       [1., 0.50, 0.5],
       [1., 0.75, 0.5]]
vel = [[0.3, 0.2, 0.1],
       [0.3, 0.2, 0.1],
       [0.3, 0.2, 0.1]]

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.FromArray(pos, vel)
u.registerParticleVector(pv=pv, ic=ic)

dumpEvery = 1

pvDump = udx.Plugins.createDumpParticles('partDump', pv, dumpEvery, [], 'h5/solvent_particles-')
u.registerPlugins(pvDump)

u.run(2)

# TEST: dump.h5.parts.mpi.2nodes
# cd dump
# rm -rf h5
# udx.run --runargs "-n 4" ./h5.parts.mpi.py > /dev/null
# udx.run h5dump h5/solvent_particles-00000.h5 > h5.parts.out.txt
