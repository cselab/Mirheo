#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse, trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--xorigin', type=float, default=0)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [8., 8., 8.]

dt   = 0.001

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

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

xobj = 0.5 * domain[0] + args.xorigin
while xobj >= domain[0]: xobj -= domain[0]

com_q = [[xobj, 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0, 0.7071, 0]]
coords = [[-2., -2., -2.],
          [ 2.,  2.,  2.]] # fake coords: don t need inside particles

m = trimesh.load(args.file);
inertia = [row[i] for i, row in enumerate(m.moment_inertia)]

mesh    = mir.ParticleVectors.Mesh(m.vertices.tolist(), m.faces.tolist())
pv_rig = mir.ParticleVectors.RigidObjectVector('rigid', mass=100, inertia=inertia, object_size=len(coords), mesh=mesh)


ic_rig = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
vv_rig = mir.Integrators.RigidVelocityVerlet("vv_rig")
u.registerParticleVector(pv_rig, ic_rig)
u.registerIntegrator(vv_rig)
u.setIntegrator(vv_rig, pv_rig)

bb = mir.Bouncers.Mesh("bounce_rig", "bounce_maxwell", kBT=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pv_rig, pv_sol)

dump_every = 500

if args.vis:
    u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv_sol, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rig, dump_every, path="ply/"))

u.registerPlugins(mir.Plugins.createDumpObjectStats("rigStats", ov=pv_rig, dump_every=dump_every, filename="stats/rigid.csv"))

u.run(5000, dt=dt)


# nTEST: bounce.rigid.mesh
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="../../../data/rbc_mesh.off"
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./mesh.py --file $f
# mir.post ../../tools/dump_csv.py stats/rigid.csv time wz qz | uscale 100 > rigid.out.txt

# nTEST: bounce.rigid.mesh.exchange
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="../../../data/rbc_mesh.off"
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./mesh.py --file $f --xorigin 4.1
# mir.post ../../tools/dump_csv.py stats/rigid.csv time wz qz | uscale 100 > rigid.out.txt
