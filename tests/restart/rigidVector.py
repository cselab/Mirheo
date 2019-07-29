#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

ranks  = args.ranks
domain = (16, 16, 16)
dt = 0.0

u = mir.mirheo(ranks, domain, dt, debug_level=9,
              log_filename='log', no_splash=True,
              checkpoint_every = (0 if args.restart else 5))

    
mesh = trimesh.creation.icosphere(subdivisions=1, radius = 0.1)

coords = [[-0.01, 0., 0.],
          [ 0.01, 0., 0.],
          [0., -0.01, 0.],
          [0.,  0.01, 0.],
          [0., 0., -0.01],
          [0., 0.,  0.01]]

udx_mesh = mir.ParticleVectors.Mesh(mesh.vertices.tolist(), mesh.faces.tolist())
pv       = mir.ParticleVectors.RigidObjectVector("pv", mass=1.0, inertia=[0.1, 0.1, 0.1], object_size=len(coords), mesh=udx_mesh)

nobjs = 10
pos = [ np.array(domain) * t for t in np.linspace(0, 1.0, nobjs) ]
Q = [ np.array([1.0, 0., 0., 0.])  for i in range(nobjs) ]
pos_q = np.concatenate((pos, Q), axis=1)

ic = mir.InitialConditions.Rigid(pos_q.tolist(), coords)

u.registerParticleVector(pv, ic)

# force correct oldMotions for correct ovStats
vv = mir.Integrators.RigidVelocityVerlet("vv")
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

if args.restart:
    u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov=pv, dump_every=5, path="stats"))

u.run(7)


# TEST: restart.rigidVector
# cd restart
# rm -rf restart stats stats.rigid*txt
# mir.run --runargs "-n 1" ./rigidVector.py --ranks 1 1 1
# mir.run --runargs "-n 2" ./rigidVector.py --ranks 1 1 1 --restart
# cat stats/pv.txt | LC_ALL=en_US.utf8 sort > stats.rigid.out.txt

# TEST: restart.rigidVector.mpi
# cd restart
# : rm -rf restart stats stats.rigid*txt
# : mir.run --runargs "-n 2" ./rigidVector.py --ranks 1 1 2
# mir.run --runargs "-n 4" ./rigidVector.py --ranks 1 1 2 --restart
# cat stats/pv.txt | LC_ALL=en_US.utf8 sort > stats.rigid.out.txt

