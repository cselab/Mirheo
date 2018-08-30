#!/usr/bin/env python

import udevicex as udx
import numpy as np
import trimesh

path   = "ply/"
pvname = "rbc"

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

mesh = udx.ParticleVectors.MembraneMesh("rbc_mesh.off")
rbc  = udx.ParticleVectors.MembraneVector(pvname, mass=1.0, mesh=mesh)
icrbc = udx.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv=rbc, ic=icrbc)

mdump = udx.Plugins.createDumpMesh("mesh_dump", rbc, 1, path)
u.registerPlugins(mdump)

u.run(3)

mesh = trimesh.load(path + pvname + "_00000.ply")

np.savetxt("vertices.txt", mesh.vertices, fmt="%g")
np.savetxt("faces.txt",    mesh.faces,    fmt="%d")

# TEST: dump.mesh
# cd dump
# rm -rf ply/ vertices.txt faces.txt mesh.out.txt 
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./mesh.py > /dev/null
# cat vertices.txt faces.txt > mesh.out.txt
