#!/usr/bin/env python

import argparse
import udevicex as udx

parser = argparse.ArgumentParser()
parser.add_argument("--sdf_file", type=str)
parser.add_argument("--domain", type=float, nargs=3)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = args.domain

density = 8

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=50.0, kbt=0.01, dt=dt, power=0.5)
u.registerInteraction(dpd)

wall = udx.Walls.SDF("sdf", args.sdf_file)
u.registerWall(wall, 0)

vv = udx.Integrators.VelocityVerlet("vv", dt)
frozen_wall = u.makeFrozenWallParticles(pvName="sdf", walls=[wall], interaction=dpd, integrator=vv, density=density)

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

# nTEST: walls.sdf.fromFile
# cd walls/sdf
# rm -rf h5
# f=../../../data/pachinko_one_post_sdf.dat
# domain=`head -n 1 $f`
# udx.run --runargs "-n 2" ./fromFile.py --sdf_file $f --domain $domain > /dev/null
# udx.avgh5 z velocity h5/solvent-0000[4-7].h5 > profile.out.txt
