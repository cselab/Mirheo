#!/usr/bin/env python3

from context import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.initialize(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

stats = udx.Plugins.createStats('stats', "stats.txt", 200)
u.registerPlugins(stats[0], stats[1])

u.run(2000)

# nTEST: dump.stats
# udx.run -n 2 ./dump.stats.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' | uscale 10 > stats.out.txt

