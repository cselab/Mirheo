#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
vtarget = (1.0, 0, 0)

density = 10

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='stdout')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', rc=1.0, a=10.0, gamma=50.0, kbt=0.1, power=0.25)
u.registerInteraction(dpd)

plate_lo = ymr.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))
plate_hi = ymr.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)

vv = ymr.Integrators.VelocityVerlet("vv")
frozen = u.makeFrozenWallParticles(pvName="frozen", walls=[plate_lo, plate_hi], interactions=[dpd], integrator=vv, density=density)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in (pv, frozen):
    u.setInteraction(dpd, p, pv)


u.registerIntegrator(vv)
u.setIntegrator(vv, pv)


gridSampleEvery = 2
gridDumpEvery   = 1000
gridBinSize     = (1., 1., 0.5)

field = ymr.Plugins.createDumpAverage('field', [pv], gridSampleEvery, gridDumpEvery, gridBinSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)


factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

vcSampleEvery = 5
vcTuneEvery = 5
vcDumpEvery = 500

vc = ymr.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, vcSampleEvery, vcTuneEvery, vcDumpEvery, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

u.run(20002)

# nTEST: flow.poiseuille
# cd flow
# rm -rf h5
# ymr.run --runargs "-n 2" ./poiseuille.py > /dev/null
# ymr.avgh5 xy velocity h5/solvent-0001[5-9].h5 | awk '{print $1}' > profile.out.txt
