#!/usr/bin/env python

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
gdot    = 0.5
tend    = 10.1

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', rc=rc, a=10.0, gamma=20.0, kbt=0.5, dt=dt, power=0.5)
u.registerInteraction(dpd)

vx = gdot*(domain[2] - 2*rc)
plate_lo = udx.Walls.Plane      ("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              rc))
plate_hi = udx.Walls.MovingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0))
u.registerWall(plate_lo, 1000)
u.registerWall(plate_hi, 1000)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_lo = u.makeFrozenWallParticles(wall=plate_lo, interaction=dpd, integrator=vv, density=density)
frozen_hi = u.makeFrozenWallParticles(wall=plate_hi, interaction=dpd, integrator=vv, density=density)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

move = udx.Integrators.Translate('move', dt=dt, velocity=(vx, 0, 0))
u.registerIntegrator(move)
u.setIntegrator(move, frozen_hi)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (8., 8., 1.0)

field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run((int)(tend/dt))

# nTEST: walls.analytic.couette
# cd walls/analytic
# rm -rf h5
# udx.run --runargs "-n 2" ./couette.py > /dev/null
# udx.avgh5 xy velocity h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt
