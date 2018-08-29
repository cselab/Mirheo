#!/usr/bin/env python

import argparse
import udevicex as udx

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["cylinder", 'sphere'])
args = parser.parse_args()


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

if   args.type == "cylinder":
    center=(domain[0]*0.5, domain[1]*0.5)
    wall = udx.Walls.Cylinder("cylinder", center=center, radius=domain[1]*0.3, axis="z", inside=False)

elif args.type == "sphere":
    center=(domain[0]*0.5, domain[1]*0.5, domain[2]*0.5)
    wall = udx.Walls.Sphere("sphere", center=center, radius=domain[1]*0.3, inside=False)

u.registerWall(wall, 0)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_wall = u.makeFrozenWallParticles(wall=wall, interaction=dpd, integrator=vv, density=density)

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

vc = udx.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 50, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.0)

field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(7002)

# nTEST: walls.analytic.obstacle.cylinder
# cd walls/analytic
# rm -rf h5
# udx.run --runargs "-n 2" ./obstacle.py --type cylinder > /dev/null
# udx.avgh5 z velocity h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt

# nTEST: walls.analytic.obstacle.sphere
# cd walls/analytic
# rm -rf h5
# udx.run --runargs "-n 2" ./obstacle.py --type sphere > /dev/null
# udx.avgh5 z velocity h5/solvent-0000[4-7].h5 | awk '{print $1}' > profile.out.txt
