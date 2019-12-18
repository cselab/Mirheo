#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes', type=float, nargs=3)
parser.add_argument('--coords', type=str)
parser.add_argument('--xorigin', type=float, default = 0)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default = [1,1,1])
args = parser.parse_args()

domain = [8., 8., 8.]

dt   = 0.001

u = mir.Mirheo(args.ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

nparts = 100
np.random.seed(42)
pos = np.random.normal(loc   = [0.5 + args.xorigin, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))

pv_sol = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic_sol = mir.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv_sol = mir.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pv_sol, ic_sol)
u.registerIntegrator(vv_sol)
u.setIntegrator(vv_sol, pv_sol)


x = 0.5 * domain[0] + args.xorigin
y = 0.5 * domain[1]
z = 0.5 * domain[2]
while x > domain[0]: x -= domain[0]
com_q = [[x, y, z,   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.vis:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= args.axes[i]
    mesh = mir.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pv_rig = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes, mesh=mesh)
else:
    pv_rig = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes)

ic_rig = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
vv_rig = mir.Integrators.RigidVelocityVerlet("ellvv")
u.registerParticleVector(pv=pv_rig, ic=ic_rig)
u.registerIntegrator(vv_rig)
u.setIntegrator(vv_rig, pv_rig)


bb = mir.Bouncers.Ellipsoid("bouncer", "bounce_back")
u.registerBouncer(bb)
u.setBouncer(bb, pv_rig, pv_sol)

dump_every = 500

if args.vis:
    u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv_sol, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rig, dump_every, path="ply/"))

u.registerPlugins(mir.Plugins.createDumpObjectStats("rigStats", pv_rig, dump_every, path="stats"))

u.run(5000)

# nTEST: bounce.rigid.ellipsoid
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=1.0; ay=2.0; az=1.0
# rm -rf pos*.txt vel*.txt
# cp ../../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./ellipsoid.py --axes $ax $ay $az --coords $f
# cat stats/ellipsoid.txt | awk '{print $2, $15, $9}' | uscale 100 > rigid.out.txt

# nTEST: bounce.rigid.ellipsoid.mpi
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=1.0; ay=2.0; az=1.0
# rm -rf pos*.txt vel*.txt
# cp ../../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 4" ./ellipsoid.py --axes $ax $ay $az --coords $f --ranks 2 1 1
# cat stats/ellipsoid.txt | awk '{print $2, $15, $9}' | uscale 100 > rigid.out.txt

# nTEST: bounce.rigid.ellipsoid.exchange
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; ax=1.0; ay=2.0; az=1.0
# rm -rf pos*.txt vel*.txt
# cp ../../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./ellipsoid.py --axes $ax $ay $az --coords $f --xorigin 4.1
# cat stats/ellipsoid.txt | awk '{print $2, $15, $9}' | uscale 100 > rigid.out.txt
