#!/usr/bin/env python

import argparse
import ymero as ymr
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

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1.0)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', rc=rc, a=12.0, gamma=gdpd, kbt=0.4, power=0.125)
u.registerInteraction(dpd)

vx = gdot*(domain[2] - 2*rc)
plate_lo = ymr.Walls.Plane      ("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              rc))
plate_hi = ymr.Walls.MovingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0))

u.registerWall(plate_lo, 1000)
u.registerWall(plate_hi, 1000)

vv = ymr.Integrators.VelocityVerlet("vv", )
frozen_lo = u.makeFrozenWallParticles(pvName="plate_lo", walls=[plate_lo], interactions=[dpd], integrator=vv, density=density)
frozen_hi = u.makeFrozenWallParticles(pvName="plate_hi", walls=[plate_hi], interactions=[dpd], integrator=vv, density=density)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

translate = ymr.Integrators.Translate('translate', velocity=(vx, 0, 0))
u.registerIntegrator(translate)
u.setIntegrator(translate, frozen_hi)

sample_every = 1
dump_every   = 1000
outname_hi = 'wallForceHi.txt'
outname_lo = 'wallForceLo.txt'
u.registerPlugins(ymr.Plugins.createWallForceCollector('forceCollectorHi', plate_hi, frozen_hi, sample_every, dump_every, outname_hi))
u.registerPlugins(ymr.Plugins.createWallForceCollector('forceCollectorLo', plate_lo, frozen_lo, sample_every, dump_every, outname_lo))

u.run((int)(tend/dt))

if u.isComputeTask():
    A = domain[0] * domain[1]
    
    data_hi = np.loadtxt(outname_hi)
    data_lo = np.loadtxt(outname_lo)

    avg_from = 5
    fx_hi = abs(np.mean(data_hi[avg_from:,1]))
    fx_lo = abs(np.mean(data_lo[avg_from:,1]))
    
    Sxy_hi = fx_hi / A
    Sxy_lo = fx_lo / A
    thSxy = abs(gdot) * 4.96 * gdpd
    print(thSxy, Sxy_hi, Sxy_lo)

# nTEST: walls.forceCollector.couette
# cd walls/
# rm -rf wallForce*txt
# ymr.run --runargs "-n 2" ./forceCollector.py  > /dev/null
# cat wallForceHi.txt | awk 'NR>1 {print $1, $2 / 10000.0, $3 / 10000.0, $4 / 10000.0}' > wallForce.out.txt
