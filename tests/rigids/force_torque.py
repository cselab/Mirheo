#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes', dest='axes', type=float, nargs=3)
parser.add_argument('--coords', dest='coords', type=str)

parser.add_argument('--const_force',  action='store_true', default=False)
parser.add_argument('--const_torque', action='store_true', default=False)
parser.add_argument('--withMesh',    action='store_true', default=False)
args = parser.parse_args()

dt   = 0.001
axes = tuple(args.axes)

ranks  = (1, 1, 1)
domain = (16, 8, 8)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.withMesh:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= axes[i]
    mesh = mir.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes, mesh=mesh)
else:
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)

ic_ell = mir.InitialConditions.Rigid(com_q, coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_ell, ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", pv_ell, dump_every=500, path="stats"))

if args.const_force:
    u.registerPlugins(mir.Plugins.createAddForce("addForce", pv_ell, force=(1., 0., 0.)))

if args.const_torque:
    u.registerPlugins(mir.Plugins.createAddTorque("addTorque", pv_ell, torque=(0., 0., 1.0)))

if args.withMesh:
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_ell, 1000, path="ply/"))

u.run(10000, dt=dt)


# nTEST: rigids.const_force
# set -eu
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./force_torque.py --axes $ax $ay $az --coords $f --const_force
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time vx comx > rigid.out.txt

# nTEST: rigids.const_torque
# set -eu
# cd rigids
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./force_torque.py --axes $ax $ay $az --coords $f --const_torque
# mir.post ../tools/dump_csv.py stats/ellipsoid.csv time wz qz > rigid.out.txt
