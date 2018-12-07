#!/usr/bin/env python

import ymero as ymr
import argparse

ranks  = (2, 1, 1)
domain = (4, 2, 2)

u = ymr.ymero(ranks, domain, debug_level=3, log_filename='log')

pos = [[1., 0.25, 0.5],
       [1., 0.50, 0.5],
       [1., 0.75, 0.5]]
vel = [[0.3, 0.2, 0.1],
       [0.3, 0.2, 0.1],
       [0.3, 0.2, 0.1]]

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.FromArray(pos, vel)
u.registerParticleVector(pv=pv, ic=ic)

dumpEvery = 1

pvDump = ymr.Plugins.createDumpParticles('partDump', pv, dumpEvery, [], 'h5/solvent_particles-')
u.registerPlugins(pvDump)

u.run(2)

# TEST: dump.h5.parts.mpi.2nodes
# cd dump
# rm -rf h5 h5.parts.out.txt
# ymr.run --runargs "-n 4" ./h5.parts.mpi.py > /dev/null
# ymr.post h5dump -d position h5/solvent_particles-00000.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8  sort > h5.parts.out.txt
