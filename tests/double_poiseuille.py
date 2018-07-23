#!/usr/bin/env python3

import sys
from context import udevicex as udx
import numpy as np

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
a = 1

u = udx.initialize(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=dt, force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

field = udx.Plugins.createDumpAverage('field', pv, sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field[0], field[1])

u.run(3002)

# nTEST: double_poiseuille
# udx.run -n 2 ./double_poiseuille.py > /dev/null
# udx.avgh5 xz velocity h5/solvent-00002.h5 | awk '{print $1}' > profile.out.txt
