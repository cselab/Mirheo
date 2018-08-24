#!/usr/bin/env python

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
vtarget = (1.0, 0, 0)

density = 10

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='stdout')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', rc=1.0, a=10.0, gamma=50.0, kbt=0.1, dt=dt, power=0.25)
u.registerInteraction(dpd)

plate_lo = udx.Walls.Plane("plate_lo", (0, 0, -1), (0, 0,              1))
plate_hi = udx.Walls.Plane("plate_hi", (0, 0,  1), (0, 0,  domain[2] - 1))
u.registerWall(plate_lo, 0)
u.registerWall(plate_hi, 0)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_lo = u.makeFrozenWallParticles(wall=plate_lo, interaction=dpd, integrator=vv, density=density)
frozen_hi = u.makeFrozenWallParticles(wall=plate_hi, interaction=dpd, integrator=vv, density=density)


u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)


for p in (pv, frozen_lo, frozen_hi):
    u.setInteraction(dpd, p, pv)


u.registerIntegrator(vv)
u.setIntegrator(vv, pv)



gridSampleEvery = 2
gridDumpEvery   = 1000
gridBinSize     = (1., 1., 0.5)

field = udx.Plugins.createDumpAverage('field', [pv], gridSampleEvery, gridDumpEvery, gridBinSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)


factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

vcSampleEvery = 5
vcDumpEvery = 500

vc = udx.Plugins.createVelocityControl("vc", "vcont.txt", pv, (0, 0, 0), domain, vcSampleEvery, vcDumpEvery, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

u.run(20002)

# nTEST: poiseuille
# cd flow
# rm -rf h5
# udx.run -n 2 ./poiseuille.py > /dev/null
# udx.avgh5 xy velocity h5/solvent-0001[5-9].h5 | awk '{print $1}' > profile.out.txt
