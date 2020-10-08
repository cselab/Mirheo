#!/usr/bin/env python

import mirheo as mir

dt   = 0.001
axes = (1, 2, 3)

ranks  = (1, 1, 1)
domain = (8, 32, 24)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv1 = mir.ParticleVectors.ParticleVector('pv1', mass = 1)
u.registerParticleVector(pv1, mir.InitialConditions.Uniform(number_density=4))

pv2 = mir.ParticleVectors.ParticleVector('pv2', mass = 1)
u.registerParticleVector(pv2, mir.InitialConditions.Uniform(number_density=4))

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=2.0, gamma=1.0, kBT=0.1, power=0.5)
u.registerInteraction(dpd)

u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=1.0, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

u.registerPlugins(mir.Plugins.createImposeVelocity('v1', pvs=[pv1],      low = (0, 0, 0), high = domain, every=1, velocity = ( 0.1, 0, 0)))
u.registerPlugins(mir.Plugins.createImposeVelocity('v2', pvs=[pv1, pv2], low = (0, 0, 0), high = domain, every=1, velocity = (-0.1, 0, 0)))

sample_every = 5
dump_every   = 100
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv2], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run(1010, dt=dt)


# nTEST: plugins.impose_velocity
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./impose_velocity.py
# mir.avgh5 yz velocities h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt
