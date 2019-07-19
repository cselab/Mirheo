#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

rc = 1.0
density = 8

u = mir.mirheo(ranks, domain, dt, no_splash=True, debug_level=0, log_filename='stdout')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.DPD('dpd', rc, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.run(50)

# TEST: log.silent
# cd log
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./silent.py > log.out.txt
# echo "end" >> log.out.txt

