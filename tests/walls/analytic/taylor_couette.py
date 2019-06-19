#!/usr/bin/env python

import argparse
import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 16)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
omega   = 0.5 # angular velocity of outer cylinder; inner is fixed
tend    = 10.1

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', rc=rc, a=10.0, gamma=20.0, kbt=0.5, power=0.5)
u.registerInteraction(dpd)

center = (domain[0]*0.5, domain[1]*0.5)
cylinder_in  = ymr.Walls.        Cylinder("cylinder_in",  center=center, radius=0.2*domain[0],    axis="z",              inside=False)
cylinder_out = ymr.Walls.RotatingCylinder("cylinder_out", center=center, radius=0.5*domain[1]-rc, axis="z", omega=omega, inside=True)

u.registerWall(cylinder_in,  1000)
u.registerWall(cylinder_out, 1000)

vv = ymr.Integrators.VelocityVerlet("vv")
frozen_in  = u.makeFrozenWallParticles(pvName="cyl_in",  walls=[cylinder_in],  interactions=[dpd], integrator=vv, density=density)
frozen_out = u.makeFrozenWallParticles(pvName="cyl_out", walls=[cylinder_out], interactions=[dpd], integrator=vv, density=density)

u.setWall(cylinder_in,  pv)
u.setWall(cylinder_out, pv)

for p in [pv, frozen_in, frozen_out]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

rotate = ymr.Integrators.Rotate('rotate', (center[0], center[1], 0.), omega=(0, 0, omega))
u.registerIntegrator(rotate)
u.setIntegrator(rotate, frozen_out)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run((int)(tend/dt))

# nTEST: walls.analytic.taylor_couette
# cd walls/analytic
# rm -rf h5
# ymr.run --runargs "-n 2" ./taylor_couette.py
# ymr.avgh5 zy velocity h5/solvent-0000[7-9].h5 > profile.out.txt
