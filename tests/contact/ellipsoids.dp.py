#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--density', dest='density', type=float)
parser.add_argument('--axes', dest='axes', type=float, nargs=3)
parser.add_argument('--coords', dest='coords', type=str)
parser.add_argument('--bounceBack', dest='bounceBack', action='store_true')
parser.set_defaults(bounceBack=False)
args = parser.parse_args()

dt   = 0.001
axes = tuple(args.axes)
a    = 0.5
density = args.density

ranks  = (1, 1, 1)
domain = (16, 8, 8)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pvSolvent = ymr.ParticleVectors.ParticleVector('solvent', mass = 1)
icSolvent = ymr.InitialConditions.Uniform(density)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.01, power=0.5)
cnt = ymr.Interactions.LJ('cnt', 1.0, epsilon=0.35, sigma=0.8, max_force=400.0)
vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction="x")

com_q = [[2.0, 5.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
         [4.0, 4.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
         [6.0, 3.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

coords = np.loadtxt(args.coords).tolist()
pvEllipsoid = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
icEllipsoid = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoid = ymr.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv=pvSolvent, ic=icSolvent)
u.registerIntegrator(vv)
u.setIntegrator(vv, pvSolvent)

u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)

u.registerInteraction(dpd)
u.registerInteraction(cnt)
u.setInteraction(dpd, pvSolvent, pvSolvent)
u.setInteraction(dpd, pvSolvent, pvEllipsoid)
u.setInteraction(cnt, pvEllipsoid, pvEllipsoid)

belongingChecker = ymr.BelongingCheckers.Ellipsoid("ellipsoidChecker")

u.registerObjectBelongingChecker(belongingChecker, pvEllipsoid)
u.applyObjectBelongingChecker(belongingChecker, pv=pvSolvent, correct_every=0, inside="none", outside="")

if args.bounceBack:
    bb = ymr.Bouncers.Ellipsoid("bounceEllipsoid")
    u.registerBouncer(bb)
    u.setBouncer(bb, pvEllipsoid, pvSolvent)

xyz = ymr.Plugins.createDumpXYZ('xyz', pvEllipsoid, 500, "xyz/")
u.registerPlugins(xyz)

ovStats = ymr.Plugins.createDumpObjectStats("objStats", ov=pvEllipsoid, dump_every=500, path="stats")
u.registerPlugins(ovStats)

u.run(10000)


# nTEST: contact.rigid.ellipsoids
# set -eu
# cd contact
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# ymr.run --runargs "-n 2" ./ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $6, $7, $8, $9}' > rigid.out.txt

# nTEST: contact.rigid.ellipsoid.bounce
# set -eu
# cd contact
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# ymr.run --runargs "-n 2" ./ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f --bounceBack > /dev/null
# cat stats/ellipsoid.txt | awk '{print $2, $6, $7, $8, $9}' > rigid.out.txt
