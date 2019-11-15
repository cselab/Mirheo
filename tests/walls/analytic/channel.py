#!/usr/bin/env python

import mirheo as mir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["squarePipe", 'cylinderPipe'])
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 8, 8)
force = (1.0, 0, 0)

density = 4
rc = 1.0

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=density)
u.registerParticleVector(pv, ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=10.0, gamma=50.0, kBT=0.01, power=0.5)
u.registerInteraction(dpd)

if   args.type == "cylinderPipe":
    center=(domain[1]*0.5, domain[2]*0.5)
    wall = mir.Walls.Cylinder("cylinder", center=center, radius=0.5*domain[1]-rc, axis="x", inside=True)

elif args.type == "squarePipe":
    lo = ( -domain[0],           rc,           rc)
    hi = (2*domain[0], domain[1]-rc, domain[2]-rc)
    wall = mir.Walls.Box("squarePipe", low=lo, high=hi, inside=True)

u.registerWall(wall, 0)

vv = mir.Integrators.VelocityVerlet("vv")
frozen_wall = u.makeFrozenWallParticles(pvName="wall", walls=[wall], interactions=[dpd], integrator=vv, number_density=density)

u.setWall(wall, pv)

for p in (pv, frozen_wall):
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor
vtarget = (0.1, 0, 0)

u.registerPlugins(mir.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 5, 50, vtarget, Kp, Ki, Kd))

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.0)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run(7002)

# nTEST: walls.analytic.channel.cylinder
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./channel.py --type cylinderPipe
# mir.avgh5 xy velocities h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt

# nTEST: walls.analytic.channel.square
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./channel.py --type squarePipe
# mir.avgh5 xy velocities h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
