#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", 200))

u.run(2000)

# nTEST: dump.stats
# cd dump
# ymr.run --runargs "-n 2" ./stats.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3 * 1e5, $4 * 1e5, $5 * 1e5}' > stats.out.txt

