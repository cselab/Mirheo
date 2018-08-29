#!/usr/bin/env python

import argparse
import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 16)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
omega   = 0.5 # angular velocity of outer cylinder; inner is fixed
tend    = 10.1

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', rc=rc, a=10.0, gamma=20.0, kbt=0.5, dt=dt, power=0.5)
u.registerInteraction(dpd)

center = (domain[0]*0.5, domain[1]*0.5)
cylinder_in  = udx.Walls.        Cylinder("cylinder_in",  center=center, radius=0.2*domain[0],    axis="z",              inside=False)
cylinder_out = udx.Walls.RotatingCylinder("cylinder_out", center=center, radius=0.5*domain[1]-rc, axis="z", omega=omega, inside=True)

u.registerWall(cylinder_in,  1000)
u.registerWall(cylinder_out, 1000)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_in  = u.makeFrozenWallParticles(wall=cylinder_in,  interaction=dpd, integrator=vv, density=density)
frozen_out = u.makeFrozenWallParticles(wall=cylinder_out, interaction=dpd, integrator=vv, density=density)

u.setWall(cylinder_in,  pv)
u.setWall(cylinder_out, pv)

for p in [pv, frozen_in, frozen_out]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

#move = udx.Integrators.Translate('move', dt=dt, velocity=(vx, 0, 0))
#u.registerIntegrator(move)
#u.setIntegrator(move, frozen_hi)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run((int)(tend/dt))

# nTEST: walls.analytic.taylorCouette
# cd walls/analytic
# rm -rf h5
# udx.run --runargs "-n 2" ./taylorCouette.py > /dev/null
# udx.avgh5 z velocity h5/solvent-0000[7-9].h5 > profile.out.txt
