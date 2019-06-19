#!/usr/bin/env python

import numpy as np
import ymero as ymr
import argparse, trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--vis', action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [8., 8., 8.]

dt   = 0.001

u = ymr.ymero(ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

nparts = 100
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pv_sol = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic_sol = ymr.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv_sol = ymr.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pv_sol, ic_sol)
u.registerIntegrator(vv_sol)
u.setIntegrator(vv_sol, pv_sol)


com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0, 0.7071, 0]]
coords = [[-2., -2., -2.],
          [ 2.,  2.,  2.]] # fake coords: don t need inside particles

m = trimesh.load(args.file);
inertia = [row[i] for i, row in enumerate(m.moment_inertia)]

mesh    = ymr.ParticleVectors.Mesh(m.vertices.tolist(), m.faces.tolist())
pv_rig = ymr.ParticleVectors.RigidObjectVector('rigid', mass=100, inertia=inertia, object_size=len(coords), mesh=mesh)


ic_rig = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
vv_rig = ymr.Integrators.RigidVelocityVerlet("vv_rig")
u.registerParticleVector(pv_rig, ic_rig)
u.registerIntegrator(vv_rig)
u.setIntegrator(vv_rig, pv_rig)

bb = ymr.Bouncers.Mesh("bounce_rig", kbt=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pv_rig, pv_sol)

dump_every = 500

if args.vis:
    u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv_sol, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(ymr.Plugins.createDumpMesh("mesh_dump", pv_rig, dump_every, path="ply/"))

u.registerPlugins(ymr.Plugins.createDumpObjectStats("rigStats", ov=pv_rig, dump_every=dump_every, path="stats"))

u.run(5000)
    

# nTEST: bounce.rigid.mesh
# set -eu
# cd bounce/rigid
# rm -rf stats rigid.out.txt
# f="../../../data/rbc_mesh.off"
# rm -rf pos*.txt vel*.txt
# ymr.run --runargs "-n 2" ./mesh.py --file $f
# cat stats/rigid.txt | awk '{print $2, $15, $9}' > rigid.out.txt
