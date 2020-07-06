#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--density',    type=float)
parser.add_argument('--axes',       type=float, nargs=3)
parser.add_argument('--coords',     type=str)
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default=(1,1,1))
args = parser.parse_args()

dt   = 0.001
axes = tuple(args.axes)
a    = 0.5
density = args.density

domain = (16, 8, 8)

nsteps = 5000

u = mir.Mirheo(args.ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True,
               checkpoint_every = (0 if args.restart else nsteps))

pv_sol = mir.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_sol = mir.InitialConditions.Uniform(density)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.01, power=0.5)
cnt = mir.Interactions.Pairwise('cnt', rc=1.0, kind="RepulsiveLJ", epsilon=0.28, sigma=0.8, max_force=400.0)
vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction="x")

com_q = [[2.0, 5.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
         [4.0, 4.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
         [6.0, 3.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

coords = np.loadtxt(args.coords).tolist()
pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv=pv_sol, ic=ic_sol)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_sol)

u.registerParticleVector(pv=pv_ell, ic=ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

u.registerInteraction(dpd)
u.registerInteraction(cnt)
u.setInteraction(dpd, pv_sol, pv_sol)
u.setInteraction(dpd, pv_sol, pv_ell)
u.setInteraction(cnt, pv_ell, pv_ell)

belongingChecker = mir.BelongingCheckers.Ellipsoid("ellipsoidChecker")

u.registerObjectBelongingChecker(belongingChecker, pv_ell)
u.applyObjectBelongingChecker(belongingChecker, pv=pv_sol, correct_every=0, inside="none", outside="")

bb = mir.Bouncers.Ellipsoid("bounce_ell", "bounce_back")
u.registerBouncer(bb)
u.setBouncer(bb, pv_ell, pv_sol)

u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv_ell, 500, [], "h5/ell"))
u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov=pv_ell, dump_every=500, path="stats"))

if args.restart:
    u.restart("restart")

u.run(nsteps)

# nTEST: restart.contact.rigid.ellipsoid
# set -eu
# cd restart
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./contact.ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f
# mir.run --runargs "-n 2" ./contact.ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f --restart
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time qw qx qy qz > rigid.out.txt

# nTEST: restart.contact.rigid.ellipsoid.mpi
# set -eu
# cd restart
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 4" ./contact.ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f --ranks 2 1 1
# mir.run --runargs "-n 4" ./contact.ellipsoids.dp.py --density $rho --axes $ax $ay $az --coords $f --ranks 2 1 1 --restart
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time qw qx qy qz > rigid.out.txt
