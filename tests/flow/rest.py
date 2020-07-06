#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

rc = 1.0
density = 8

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', "stats", 1000))

u.run(5001)

# nTEST: flow.rest
# cd flow
# rm -rf stats.csv
# mir.run --runargs "-n 2" ./rest.py > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt
