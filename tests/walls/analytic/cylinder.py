#!/usr/bin/env python

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 8, 8)
force = (1.0, 0, 0)

density = 4

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='stdout')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=50.0, kbt=0.01, dt=dt, power=0.5)
u.registerInteraction(dpd)

center=(domain[0]*0.5, domain[1]*0.5)
cylinder = udx.Walls.Cylinder("cylinder", center=center, radius=domain[1]*0.3, axis="z", inside=False)
u.registerWall(cylinder, 0)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_cylinder = u.makeFrozenWallParticles(wall=cylinder, interaction=dpd, integrator=vv, density=density)

u.setWall(cylinder, pv)

for p in (pv, frozen_cylinder):
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor
vtarget = (0.1, 0, 0)

vc = udx.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 50, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.0)

field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(7002)

# nTEST: walls.analytic.cylinder
# cd walls/analytic
# rm -rf h5
# udx.run --runargs "-n 2" ./cylinder.py > /dev/null
# udx.avgh5 z velocity h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
