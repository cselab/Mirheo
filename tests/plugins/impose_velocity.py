#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

dt   = 0.001
axes = (1, 2, 3)

ranks  = (1, 1, 1)
domain = (8, 32, 24)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv1 = ymr.ParticleVectors.ParticleVector('pv1', mass = 1)
u.registerParticleVector(pv1, ymr.InitialConditions.Uniform(density=4))
    
pv2 = ymr.ParticleVectors.ParticleVector('pv2', mass = 1)
u.registerParticleVector(pv2, ymr.InitialConditions.Uniform(density=4))
    
dpd = ymr.Interactions.DPD('dpd', 1.0, a=2.0, gamma=1.0, kbt=0.1, power=0.5)
u.registerInteraction(dpd)

u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)
    
vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=1.0, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

u.registerPlugins(ymr.Plugins.createImposeVelocity('v1', pvs=[pv1],      low = (0, 0, 0), high = domain, every=1, velocity = ( 0.1, 0, 0)))
u.registerPlugins(ymr.Plugins.createImposeVelocity('v2', pvs=[pv1, pv2], low = (0, 0, 0), high = domain, every=1, velocity = (-0.1, 0, 0)))

sample_every = 5
dump_every   = 100
bin_size     = (1., 1., 1.)

field = ymr.Plugins.createDumpAverage('field', [pv2], sample_every, dump_every, bin_size, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(1010)


# nTEST: plugins.imposeVelocity
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 2" ./impose_velocity.py > /dev/null
# ymr.avgh5 yz velocity h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt
