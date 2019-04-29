#!/usr/bin/env python

import ymero as ymr
import numpy as np
import trimesh, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (12, 8, 10)

rc=1.0

u = ymr.ymero(ranks, domain, dt=0, debug_level=8, log_filename='log', no_splash=True)

m = trimesh.load(args.mesh);
mesh = ymr.ParticleVectors.MembraneMesh(m.vertices.tolist(), m.faces.tolist())

rbc  = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh)
icrbc = ymr.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv=rbc, ic=icrbc)

box_lo = (          - domain[0],           - domain[1],             rc)
box_hi = (domain[0] + domain[0], domain[1] + domain[1], domain[2] - rc)
plates = ymr.Walls.Box("plates", box_lo, box_hi, inside=True)
u.registerWall(plates, 0)

# fake repulsion module to force sdf computation
u.registerPlugins(ymr.Plugins.createWallRepulsion("wallRepulsion", rbc, plates, C=500, h=rc, max_force=500))

dump_every = 1
u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh('partDump', rbc, dump_every, [["sdf", "scalar"]], 'h5/rbc-'))

u.run(2)

# nTEST: dump.h5.mesh.sdf
# cd dump
# rm -rf h5
# mesh="../../data/rbc_mesh.off"
# ymr.run --runargs "-n 2" ./h5.mesh.sdf.py --mesh $mesh
# ymr.post h5dump -d sdf h5/rbc-00001.h5 | awk '{print $2}' > h5.mesh.sdf.out.txt
