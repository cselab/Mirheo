#!/usr/bin/env python

import ymero as ymr
import numpy as np
import trimesh, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

m = trimesh.load(args.mesh);
mesh = ymr.ParticleVectors.MembraneMesh(m.vertices.tolist(), m.faces.tolist())

pv_rbc = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh)
ic_rbc = ymr.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

dumpEvery = 1
u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh('partDump', pv_rbc, dumpEvery, [], 'h5/rbc-'))

u.run(2)

# TEST: dump.h5.mesh
# cd dump
# rm -rf h5
# mesh="../../data/rbc_mesh.off"
# ymr.run --runargs "-n 2" ./h5.mesh.py --mesh $mesh
# ymr.post h5dump -d position h5/rbc-00000.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8 sort > h5.mesh.out.txt
