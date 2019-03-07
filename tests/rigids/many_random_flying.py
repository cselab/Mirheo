#!/usr/bin/env python

import ymero as ymr
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

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

np.random.seed(84)
com_q = np.random.rand(args.nobjects, 7)
com_q[:, 0:3] = np.multiply(com_q[:, 0:3], np.array(domain)) 
vels  = np.random.rand(args.nobjects, 3)

coords = np.loadtxt(args.coords).tolist()


pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
icEllipsoid = ymr.InitialConditions.Rigid(com_q=com_q.tolist(), coords=coords, init_vels=vels.tolist())
vvEllipsoid = ymr.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)

u.registerPlugins( ymr.Plugins.createDumpObjectStats("objStats", ov=pvEllipsoid, dump_every=10, path="stats") )
u.run(1000)


# nTEST: freefly.onerank
# set -eu
# cd rigids
# f="pos.txt"
# rm -rf stats freefly.out.txt $f
# rho=4.0; ax=1.0; ay=2.0; az=3.0
# ymr.run --runargs "-n 2"  ./createEllipsoid.py --density $rho --axes $ax $ay $az --out $f --niter 1000  > /dev/null
# ymr.run --runargs "-n 2" ./many_random_flying.py --nranks 1 1 1 --nobjects 55  $f > /dev/null
# LC_ALL=en_US.utf8 sort -g -k1 -k2 stats/ellipsoid.txt > freefly.out.txt

# nTEST: freefly.manyranks
# set -eu
# cd rigids
# f="pos.txt"
# rm -rf stats freefly.out.txt $f
# rho=4.0; ax=1.0; ay=2.0; az=3.0
# ymr.run --runargs "-n 2"  ./createEllipsoid.py --density $rho --axes $ax $ay $az --out $f --niter 1000  > /dev/null
# ymr.run --runargs "-n 12" ./many_random_flying.py --nranks 1 2 3 --nobjects 123  $f > /dev/null
# LC_ALL=en_US.utf8 sort -g -k1 -k2 stats/ellipsoid.txt > freefly.out.txt
