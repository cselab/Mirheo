#!/usr/bin/env python

import sys, argparse
import numpy as np
import mirheo as mir

tend = 1.0
nsteps = 500
dt = tend/nsteps

L = 8
ranks  = (1, 1, 1)
domain = (L, L, L)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

nparts = 10
pos = np.full(shape=(nparts, 3),
              fill_value=[L/2, L/2, L/2])

vel = np.zeros_like(pos)
vel[:,0] = np.linspace(0, 1, nparts)


pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv = mir.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pv, ic)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

dump_every = nsteps//5
u.registerPlugins(mir.Plugins.createParticleChannelAverager('avg', pv, "velocities", "avg_vel", updateEvery=dump_every))
u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv, dump_every, ["avg_vel"], 'h5/pv-'))

u.run(nsteps, dt=dt)


# nTEST: plugins.particles_average
# set -eu
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./particles_average.py
# mir.post h5dump -d avg_vel h5/pv-00003.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8  sort > avg_vel.out.txt
