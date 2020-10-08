#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default=(1,1,1))
args = parser.parse_args()

dt = 0.001

domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
gdot    = 0.5 # shear rate
T       = 3.0 # period for oscillating plate case
tend    = 5.0

nsteps = int (tend / dt)

u = mir.Mirheo(args.ranks, domain, debug_level=3, log_filename='log', no_splash=True,
               checkpoint_every = (0 if args.restart else nsteps))

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=rc, kind="DPD", a=10.0, gamma=20.0, kBT=1.0, power=0.125)
u.registerInteraction(dpd)

vx = gdot*(domain[2] - 2*rc)
plate_lo = mir.Walls.Plane      ("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              rc))
plate_hi = mir.Walls.MovingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0))

nsteps_eq = (1 if args.restart else 1000)
u.registerWall(plate_lo, nsteps_eq)
u.registerWall(plate_hi, nsteps_eq)

vv = mir.Integrators.VelocityVerlet("vv", )
frozen_lo = u.makeFrozenWallParticles(pvName="plate_lo", walls=[plate_lo], interactions=[dpd], integrator=vv, number_density=density, dt=dt)
frozen_hi = u.makeFrozenWallParticles(pvName="plate_hi", walls=[plate_hi], interactions=[dpd], integrator=vv, number_density=density, dt=dt)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

move = mir.Integrators.Translate('move', velocity=(vx, 0, 0))
u.registerIntegrator(move)
u.setIntegrator(move, frozen_hi)

sample_every = 2
dump_every   = 1000
bin_size     = (8., 8., 1.0)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

if args.restart:
    u.restart("restart")

u.run(nsteps + 1, dt=dt)

# nTEST: restart.walls
# cd restart
# rm -rf h5
# mir.run --runargs "-n 2" ./walls.py
# mir.run --runargs "-n 2" ./walls.py --restart
# mir.avgh5 xy velocities h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt

# nTEST: restart.walls.mpi
# cd restart
# rm -rf h5
# mir.run --runargs "-n 4" ./walls.py --ranks 1 2 1
# mir.run --runargs "-n 4" ./walls.py --ranks 1 2 1 --restart
# mir.avgh5 xy velocities h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt
