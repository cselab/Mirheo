#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nranks', type=int, nargs=3, default=[1,1,1])
parser.add_argument('--nobjects', type=int, default=10)
parser.add_argument('coords', type=str)

args = parser.parse_args()

dt   = 0.5
axes = (1, 2, 3)

ranks  = tuple(args.nranks)
domain = (31, 18, 59)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

np.random.seed(84)
com_q = np.random.rand(args.nobjects, 7)
com_q[:, 0:3] = np.multiply(com_q[:, 0:3], np.array(domain))
vels  = np.random.rand(args.nobjects, 3)

coords = np.loadtxt(args.coords).tolist()

pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = mir.InitialConditions.Rigid(com_q=com_q.tolist(), coords=coords, init_vels=vels.tolist())
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_ell, ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

u.registerPlugins( mir.Plugins.createDumpObjectStats("objStats", pv_ell, dump_every=10, path="stats") )
u.run(1000, dt=dt)


# nTEST: rigids.many_random_flying.onerank
# set -eu
# cd rigids
# f="pos.txt"
# rm -rf stats freefly.out.txt $f
# rho=4.0; ax=1.0; ay=2.0; az=3.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./many_random_flying.py --nranks 1 1 1 --nobjects 55  $f
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv objId time comx comy comz qw qx qy qz vx vy vz wx wy wz fx fy fz Tx Ty Tz | LC_ALL=en_US.utf8 sort -g -k1 -k2 > freefly.out.txt

# nTEST: rigids.many_random_flying.manyranks
# set -eu
# cd rigids
# f="pos.txt"
# rm -rf stats freefly.out.txt $f
# rho=4.0; ax=1.0; ay=2.0; az=3.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 12" ./many_random_flying.py --nranks 1 2 3 --nobjects 123  $f
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv objId time comx comy comz qw qx qy qz vx vy vz wx wy wz fx fy fz Tx Ty Tz | LC_ALL=en_US.utf8 sort -g -k1 -k2 > freefly.out.txt
