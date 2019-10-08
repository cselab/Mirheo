#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument("--sdf_file", type=str)
parser.add_argument("--domain", type=float, nargs=3)
parser.add_argument("--vtarget", type=int, default=0.1)
parser.add_argument("--niters", type=int, default=7002)

args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = args.domain

density = 8

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=50.0, kbt=0.01, power=0.5)
u.registerInteraction(dpd)

wall = mir.Walls.SDF("sdf", args.sdf_file)
u.registerWall(wall, 100)
u.dumpWalls2XDMF([wall], (0.5, 0.5, 0.5), filename='h5/wall')


vv = mir.Integrators.VelocityVerlet("vv")
frozen_wall = u.makeFrozenWallParticles(pvName="sdf", walls=[wall], interactions=[dpd], integrator=vv, density=density)

u.setWall(wall, pv)

for p in (pv, frozen_wall):
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor
vtarget = (args.vtarget, 0, 0)

u.registerPlugins(mir.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 5, 50, vtarget, Kp, Ki, Kd))

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.0)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run(args.niters)

# nTEST: walls.sdf.from_file.profile
# cd walls/sdf
# rm -rf h5
# f=../../../data/pachinko_one_post_sdf.dat
# domain=`head -n 1 $f`
# mir.run --runargs "-n 2" ./from_file.py --sdf_file $f --domain $domain
# mir.avgh5 z velocity h5/solvent-0000[4-7].h5 > profile.out.txt

# nTEST: walls.sdf.from_file.sdf
# cd walls/sdf
# rm -rf h5
# f=../../../data/pachinko_one_post_sdf.dat
# domain=`head -n 1 $f`
# mir.run --runargs "-n 2" ./from_file.py --sdf_file $f --domain $domain --niters=0
# mir.avgh5 z sdf h5/wall.h5 > sdf.out.txt

# nTEST: walls.sdf.from_file.particles
# cd walls/sdf
# rm -rf h5
# f=../../../data/pachinko_one_post_sdf.dat
# domain=`head -n 1 $f`
# mir.run --runargs "-n 2" ./from_file.py --sdf_file $f --domain $domain --niters=5002 --vtarget=5
# grep "inside the wall" log_00000.log | awk '{print $6 / 100.0;}' > particles.out.txt
