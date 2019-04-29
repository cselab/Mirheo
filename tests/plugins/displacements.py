#!/usr/bin/env python

import ymero as ymr
import argparse
import numpy as np

ranks  = (1, 1, 1)
domain = (8, 8, 8)

dt = 0.01

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

n = 20
np.random.seed(42)
positions  = np.random.rand(n, 3)
velocities = np.random.rand(n, 3) - 0.5

for i in range(3):
    positions[:,i] *= domain[i]

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.FromArray(positions.tolist(), velocities.tolist())
u.registerParticleVector(pv=pv, ic=ic)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

dump_every = 20
update_every = dump_every

u.registerPlugins(ymr.Plugins.createParticleDisplacement('disp', pv, update_every))
u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, dump_every, [["displacements", "vector"]], 'h5/solvent_particles-'))

u.run(100)

# TEST: plugins.displacements
# cd plugins
# rm -rf h5 displacements.out.txt
# ymr.run --runargs "-n 2" ./displacements.py
# ymr.post h5dump -d displacements h5/solvent_particles-00004.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8 sort > displacements.out.txt
