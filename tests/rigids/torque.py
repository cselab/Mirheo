#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes', dest='axes', type=float, nargs=3)
parser.add_argument('--coords', dest='coords', type=str)
args = parser.parse_args()

dt   = 0.001
axes = tuple(args.axes)

ranks  = (1, 1, 1)
domain = (16, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()
pvEllipsoid = udx.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
icEllipsoid = udx.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoid = udx.Integrators.RigidVelocityVerlet("ellvv", dt)

u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)


xyz = udx.Plugins.createDumpXYZ('xyz', pvEllipsoid, 500, "xyz/")
u.registerPlugins(xyz)

ovStats = udx.Plugins.createDumpObjectStats("objStats", ov=pvEllipsoid, dump_every=500, path="stats")
u.registerPlugins(ovStats)

addTorque = udx.Plugins.createAddTorque("addTorque", pvEllipsoid, torque=(0., 0., 1.0))
u.registerPlugins(addTorque)

u.run(10000)


# nTEST: rigids.torque
# set -eu
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# common_args="--axes 2.0 1.0 1.0"
# udx.run ./createEllipsoid.py $common_args --density 8 --out $f --niter 1000  > /dev/null
# udx.run --runargs "-n 2" ./torque.py $common_args --coords $f > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $15, $9}' > rigid.out.txt
