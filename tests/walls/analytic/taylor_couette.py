#!/usr/bin/env python

import argparse
import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 16)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
omega   = 0.5 # angular velocity of outer cylinder; inner is fixed
tend    = 10.1

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=10.0, gamma=20.0, kBT=0.5, power=0.5)
u.registerInteraction(dpd)

center = (domain[0]*0.5, domain[1]*0.5)
cylinder_in  = mir.Walls.        Cylinder("cylinder_in",  center=center, radius=0.2*domain[0],    axis="z",              inside=False)
cylinder_out = mir.Walls.RotatingCylinder("cylinder_out", center=center, radius=0.5*domain[1]-rc, axis="z", omega=omega, inside=True)

u.registerWall(cylinder_in,  1000)
u.registerWall(cylinder_out, 1000)

vv = mir.Integrators.VelocityVerlet("vv")
frozen_in  = u.makeFrozenWallParticles(pvName="cyl_in",  walls=[cylinder_in],  interactions=[dpd], integrator=vv, number_density=density)
frozen_out = u.makeFrozenWallParticles(pvName="cyl_out", walls=[cylinder_out], interactions=[dpd], integrator=vv, number_density=density)

u.setWall(cylinder_in,  pv)
u.setWall(cylinder_out, pv)

for p in [pv, frozen_in, frozen_out]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

rotate = mir.Integrators.Rotate('rotate', (center[0], center[1], 0.), omega=(0, 0, omega))
u.registerIntegrator(rotate)
u.setIntegrator(rotate, frozen_out)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run((int)(tend/dt))

# nTEST: walls.analytic.taylor_couette
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./taylor_couette.py
# mir.avgh5 zy velocities h5/solvent-0000[7-9].h5 > profile.out.txt
