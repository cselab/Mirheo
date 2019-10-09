#!/usr/bin/env python

import mirheo as mir
import numpy as np
import trimesh, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--readFrom", choices=["off", 'trimesh'])
args = parser.parse_args()


path   = "ply/"
pvname = "rbc"
off    = "rbc_mesh.off"

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

if args.readFrom == "off":
    mesh = mir.ParticleVectors.MembraneMesh(off)
elif args.readFrom == "trimesh":
    m = trimesh.load(off);
    mesh = mir.ParticleVectors.MembraneMesh(m.vertices.tolist(), m.faces.tolist())

pv_rbc = mir.ParticleVectors.MembraneVector(pvname, mass=1.0, mesh=mesh)
ic_rbc = mir.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, 1, path))

u.run(3)

if u.isMasterTask():
    mesh = trimesh.load(path + pvname + "_00000.ply")
    np.savetxt("vertices.txt", mesh.vertices, fmt="%g")
    np.savetxt("faces.txt",    mesh.faces,    fmt="%d")

# TEST: dump.mesh
# cd dump
# rm -rf ply/ vertices.txt faces.txt mesh.out.txt 
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mesh.py --readFrom off
# cat vertices.txt faces.txt > mesh.out.txt

# TEST: dump.mesh.fromTrimesh
# cd dump
# rm -rf ply/ vertices.txt faces.txt mesh.out.txt 
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mesh.py --readFrom trimesh
# cat vertices.txt faces.txt > mesh.out.txt
