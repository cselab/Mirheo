#!/usr/bin/env python3

import sys
sys.path.insert(0, "../..")
from common.context import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)


vv = udx.Integrators.VelocityVerlet_withConstForce("vv", dt, force)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

plate_lo = udx.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
plate_hi = udx.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))

u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)
u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 0.5)

field = udx.Plugins.createDumpAverage('field', pv, sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(7002)

# nTEST: analytic.plates
# cd walls/analytic
# udx.run --runargs "-n 2" ./plates.py > /dev/null
# udx.avgh5 xy velocity h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
