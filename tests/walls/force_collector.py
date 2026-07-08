#!/usr/bin/env python

import mirheo as mir
import numpy as np

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 10
rc      = 1.0
gdot    = -0.5 # shear rate
tend    = 10.1

gdpd = 11.0

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1.0)
ic = mir.InitialConditions.Uniform(number_density=density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=12.0, gamma=gdpd, kBT=0.4, power=0.125)
u.registerInteraction(dpd)

vx = gdot*(domain[2] - 2*rc)
plate_lo = mir.Walls.Plane      ("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              rc))
plate_hi = mir.Walls.MovingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0))

u.registerWall(plate_lo, 1000)
u.registerWall(plate_hi, 1000)

vv = mir.Integrators.VelocityVerlet("vv", )
frozen_lo = u.makeFrozenWallParticles(pvName="plate_lo", walls=[plate_lo], interactions=[dpd], integrator=vv, number_density=density, dt=dt)
frozen_hi = u.makeFrozenWallParticles(pvName="plate_hi", walls=[plate_hi], interactions=[dpd], integrator=vv, number_density=density, dt=dt)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

translate = mir.Integrators.Translate('translate', velocity=(vx, 0, 0))
u.registerIntegrator(translate)
u.setIntegrator(translate, frozen_hi)

sample_every = 1
dump_every   = 1000
outname_hi = 'wallForceHi.csv'
outname_lo = 'wallForceLo.csv'
u.registerPlugins(mir.Plugins.createWallForceCollector('forceCollectorHi', plate_hi, frozen_hi, sample_every, dump_every, outname_hi))
u.registerPlugins(mir.Plugins.createWallForceCollector('forceCollectorLo', plate_lo, frozen_lo, sample_every, dump_every, outname_lo))

u.run(int(tend / dt), dt=dt)


# nTEST: walls.force_collector.couette
# cd walls/
# rm -rf wallForce*txt
# mir.run --runargs "-n 2" ./force_collector.py
# mir.post ../tools/dump_csv.py wallForceHi.csv time fx fy fz | awk 'NR>1 {print $1, $2 / 10000.0, $3 / 10000.0, $4 / 10000.0}' > wallForce.out.txt
