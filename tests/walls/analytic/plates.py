#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 4

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=50.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)

plate_lo = ymr.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))
plate_hi = ymr.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)

vv = ymr.Integrators.VelocityVerlet("vv")
frozen = u.makeFrozenWallParticles(pvName="plates", walls=[plate_lo, plate_hi], interactions=[dpd], integrator=vv, density=density)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)


for p in (pv, frozen):
    u.setInteraction(dpd, p, pv)

vv_dp = ymr.Integrators.VelocityVerlet_withConstForce("vv_dp", force)
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv)


sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 0.5)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float4")], 'h5/solvent-')
u.registerPlugins(field)

#u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", 1000))

u.run(7002)

# nTEST: walls.analytic.plates
# cd walls/analytic
# rm -rf h5
# ymr.run --runargs "-n 2" ./plates.py > /dev/null
# ymr.avgh5 xy velocity h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
