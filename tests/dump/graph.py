#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (4, 4, 4)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.run(1)

u.save_dependency_graph_graphml("tasks")

# nTEST: dump.graph
# cd dump
# rm -rf tasks.graphml
# ymr.run --runargs "-n 1" ./graph.py > /dev/null
# cat tasks.graphml > tasks.out.txt

