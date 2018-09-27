#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

ranks  = args.ranks
domain = (16, 16, 16)

if args.restart:
    u = udx.udevicex(ranks, domain, debug_level=8, log_filename='log', checkpoint_every=0)
else:
    u = udx.udevicex(ranks, domain, debug_level=8, log_filename='log', checkpoint_every=5)

    
mesh = trimesh.creation.icosphere(subdivisions=1, radius = 0.1)

coords = [[-0.01, 0., 0.],
          [ 0.01, 0., 0.],
          [0., -0.01, 0.],
          [0.,  0.01, 0.],
          [0., 0., -0.01],
          [0., 0.,  0.01]]

udx_mesh = udx.ParticleVectors.Mesh(mesh.vertices.tolist(), mesh.faces.tolist())
pv       = udx.ParticleVectors.RigidObjectVector("pv", mass=1.0, inertia=[0.1, 0.1, 0.1], object_size=len(coords), mesh=udx_mesh)

if args.restart:
    ic   = udx.InitialConditions.Restart("restart/")
else:
    nobjs = 10
    pos = [ np.array(domain) * t for t in np.linspace(0, 1.0, nobjs) ]
    Q = [ np.array([1.0, 0., 0., 0.])  for i in range(nobjs) ]
    pos_q = np.concatenate((pos, Q), axis=1)

    ic = udx.InitialConditions.Rigid(pos_q.tolist(), coords)

u.registerParticleVector(pv, ic)

if args.restart:
    stats = udx.Plugins.createDumpObjectStats("objStats", ov=pv, dump_every=5, path="stats")
    u.registerPlugins(stats)

u.run(7)
    

# TEST: restart.rigidVector
# cd restart
# rm -rf restart stats stats.rigid*txt
# udx.run --runargs "-n 1" ./rigidVector.py --ranks 1 1 1           > /dev/null
# udx.run --runargs "-n 2" ./rigidVector.py --ranks 1 1 1 --restart > /dev/null
# cat stats/pv.txt | sort > stats.rigid.out.txt

# TEST: restart.rigidVector.mpi
# cd restart
# rm -rf restart stats stats.rigid*txt
# udx.run --runargs "-n 2" ./rigidVector.py --ranks 1 1 2           > /dev/null
# udx.run --runargs "-n 4" ./rigidVector.py --ranks 1 1 2 --restart > /dev/null
# cat stats/pv.txt | sort > stats.rigid.out.txt

