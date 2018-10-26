#!/usr/bin/env python

import udevicex as udx
import numpy as np
import trimesh, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = udx.udevicex(ranks, domain, debug_level=8, log_filename='log')

m = trimesh.load(args.mesh);
mesh = udx.ParticleVectors.MembraneMesh(m.vertices.tolist(), m.faces.tolist())

rbc  = udx.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh)
icrbc = udx.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv=rbc, ic=icrbc)

dumpEvery = 1
ovDump = udx.Plugins.createDumpParticlesWithMesh('partDump', rbc, dumpEvery, [], 'h5/rbc-')
u.registerPlugins(ovDump)

u.run(2)

# TEST: dump.h5.mesh
# cd dump
# rm -rf h5
# mesh="../../data/rbc_mesh.off"
# udx.run --runargs "-n 2" ./h5.mesh.py --mesh $mesh > /dev/null
# udx.post h5dump -d position h5/rbc-00000.h5 | awk '{print $2, $3, $4}' | sort > h5.mesh.out.txt
