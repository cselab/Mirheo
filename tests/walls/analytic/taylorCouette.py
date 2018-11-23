#!/usr/bin/env python

import argparse
import udevicex as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 16)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
omega   = 0.5 # angular velocity of outer cylinder; inner is fixed
tend    = 10.1

u = ymr.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', rc=rc, a=10.0, gamma=20.0, kbt=0.5, dt=dt, power=0.5)
u.registerInteraction(dpd)

center = (domain[0]*0.5, domain[1]*0.5)
cylinder_in  = ymr.Walls.        Cylinder("cylinder_in",  center=center, radius=0.2*domain[0],    axis="z",              inside=False)
cylinder_out = ymr.Walls.RotatingCylinder("cylinder_out", center=center, radius=0.5*domain[1]-rc, axis="z", omega=omega, inside=True)

u.registerWall(cylinder_in,  1000)
u.registerWall(cylinder_out, 1000)

vv = ymr.Integrators.VelocityVerlet("vv", dt)
frozen_in  = u.makeFrozenWallParticles(pvName="cyl_in",  walls=[cylinder_in],  interaction=dpd, integrator=vv, density=density)
frozen_out = u.makeFrozenWallParticles(pvName="cyl_out", walls=[cylinder_out], interaction=dpd, integrator=vv, density=density)

u.setWall(cylinder_in,  pv)
u.setWall(cylinder_out, pv)

for p in [pv, frozen_in, frozen_out]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

rotate = ymr.Integrators.Rotate('rotate', dt, (center[0], center[1], 0.), omega=(0, 0, omega))
u.registerIntegrator(rotate)
u.setIntegrator(rotate, frozen_out)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run((int)(tend/dt))

# nTEST: walls.analytic.taylorCouette
# cd walls/analytic
# rm -rf h5
# ymr.run --runargs "-n 2" ./taylorCouette.py > /dev/null
# ymr.avgh5 zy velocity h5/solvent-0000[7-9].h5 > profile.out.txt
