#!/usr/bin/env python

import udevicex as udx
import numpy as np

ranks  = (1, 1, 1)
domain = (2, 2, 4)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=3)
u.registerParticleVector(pv=pv, ic=ic)

dumpEvery   = 1
binSize     = (1., 1., 1.)

pvDump = udx.Plugins.createDumpParticles('partDump', pv, dumpEvery, [], 'h5/solvent_particles-')
u.registerPlugins(pvDump)

u.run(2)

# TEST: dump.h5.parts
# cd dump
# rm -rf h5
# udx.run --runargs "-n 2" ./h5.parts.py > /dev/null
# h5dump h5/solvent_particles-00000.h5 > h5.parts.out.txt
