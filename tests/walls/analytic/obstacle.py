#!/usr/bin/env python

import argparse
import ymero as ymr

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["cylinder", 'sphere'])
args = parser.parse_args()


dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 8, 8)
force = (1.0, 0, 0)

density = 4

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='stdout')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=50.0, kbt=0.01, power=0.5)
u.registerInteraction(dpd)

if   args.type == "cylinder":
    center=(domain[0]*0.5, domain[1]*0.5)
    wall = ymr.Walls.Cylinder("cylinder", center=center, radius=domain[1]*0.3, axis="z", inside=False)

elif args.type == "sphere":
    center=(domain[0]*0.5, domain[1]*0.5, domain[2]*0.5)
    wall = ymr.Walls.Sphere("sphere", center=center, radius=domain[1]*0.3, inside=False)

u.registerWall(wall, 0)

vv = ymr.Integrators.VelocityVerlet("vv")
frozen_wall = u.makeFrozenWallParticles(pvName="wall", walls=[wall], interactions=[dpd], integrator=vv, density=density)

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

vc = ymr.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 5, 50, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.0)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(7002)

# nTEST: walls.analytic.obstacle.cylinder
# cd walls/analytic
# rm -rf h5
# ymr.run --runargs "-n 2" ./obstacle.py --type cylinder > /dev/null
# ymr.avgh5 z velocity h5/solvent-0000[4-7].h5 > profile.out.txt

# nTEST: walls.analytic.obstacle.sphere
# cd walls/analytic
# rm -rf h5
# ymr.run --runargs "-n 2" ./obstacle.py --type sphere > /dev/null
# ymr.avgh5 z velocity h5/solvent-0000[4-7].h5 > profile.out.txt
