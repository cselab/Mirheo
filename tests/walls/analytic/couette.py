#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["stationary", 'oscillating'])
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 16, 8)
force = (1.0, 0, 0)

density = 8
rc      = 1.0
gdot    = 0.5 # shear rate
T       = 3.0 # period for oscillating plate case
tend    = 10.1

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = mir.Interactions.DPD('dpd', rc=rc, a=10.0, gamma=20.0, kbt=1.0, power=0.125)
u.registerInteraction(dpd)

vx = gdot*(domain[2] - 2*rc)
plate_lo = mir.Walls.Plane      ("plate_lo", normal=(0, 0, -1), pointThrough=(0, 0,              rc))

if args.type == "oscillating":
    plate_hi = mir.Walls.OscillatingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0), period=T)
else:
    plate_hi = mir.Walls.MovingPlane("plate_hi", normal=(0, 0,  1), pointThrough=(0, 0,  domain[2] - rc), velocity=(vx, 0, 0))

u.registerWall(plate_lo, 1000)
u.registerWall(plate_hi, 1000)

vv = mir.Integrators.VelocityVerlet("vv", )
frozen_lo = u.makeFrozenWallParticles(pvName="plate_lo", walls=[plate_lo], interactions=[dpd], integrator=vv, density=density)
frozen_hi = u.makeFrozenWallParticles(pvName="plate_hi", walls=[plate_hi], interactions=[dpd], integrator=vv, density=density)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

if args.type == 'oscillating':
    move = mir.Integrators.Oscillate('osc', velocity=(vx, 0, 0), period=T)
else:
    move = mir.Integrators.Translate('move', velocity=(vx, 0, 0))
u.registerIntegrator(move)
u.setIntegrator(move, frozen_hi)

sample_every = 2
dump_every   = 1000
bin_size     = (8., 8., 1.0)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run((int)(tend/dt))

# nTEST: walls.analytic.couette
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./couette.py --type stationary
# mir.avgh5 xy velocity h5/solvent-0000[7-9].h5 | awk '{print $1}' > profile.out.txt

# nTEST: walls.analytic.couette.oscillating
# cd walls/analytic
# rm -rf h5
# mir.run --runargs "-n 2" ./couette.py --type oscillating
# mir.avgh5 xy velocity h5/solvent-00009.h5 | awk '{print $1}' > profile.out.txt
