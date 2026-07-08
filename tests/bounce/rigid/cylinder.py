#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--coords', type=str)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [8., 8., 8.]

radius = 1.5
length = 5.0

dt   = 0.001

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

nparts = 100
np.random.seed(42)
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2] + 1.5],
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


com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.vis:
    import trimesh
    cyl_mesh = trimesh.creation.cylinder(radius = radius, height = length)
    mesh = mir.ParticleVectors.Mesh(cyl_mesh.vertices.tolist(), cyl_mesh.faces.tolist())
    ov_rig = mir.ParticleVectors.RigidCylinderVector('cylinder', mass=1, object_size=len(coords), radius=radius, length=length, mesh=mesh)
else:
    ov_rig = mir.ParticleVectors.RigidCylinderVector('cylinder', mass=1, object_size=len(coords), radius=radius, length=length)

ic_rig = mir.InitialConditions.Rigid(com_q, coords)
vv_rig = mir.Integrators.RigidVelocityVerlet("cylvv")
u.registerParticleVector(ov_rig, ic_rig)
u.registerIntegrator(vv_rig)
u.setIntegrator(vv_rig, ov_rig)


bb = mir.Bouncers.Cylinder("bouncer", "bounce_back")
u.registerBouncer(bb)
u.setBouncer(bb, ov_rig, pv_sol)

dump_every = 500

if args.vis:
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv_sol, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", ov_rig, dump_every, path="ply/"))

u.registerPlugins(mir.Plugins.createDumpObjectStats("rigStats", ov_rig, dump_every, filename="stats/cylinder.csv"))

u.run(5000, dt=dt)

# nTEST: bounce.rigid.cylinder
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="pos.txt"
# rho=8.0; R=1.5; L=5.0
# rm -rf pos*.txt vel*.txt
# cp ../../../data/cylinder_coords_${rho}_${R}_${L}.txt $f
# mir.run --runargs "-n 2" ./cylinder.py --coords $f
# mir.post ../../tools/dump_csv.py stats/cylinder.csv time wz qz | uscale 100 > rigid.out.txt
