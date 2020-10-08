#!/usr/bin/env python

import mirheo as mir
import trimesh, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (12, 8, 10)

rc=1.0

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

m = trimesh.load(args.mesh);
mesh = mir.ParticleVectors.MembraneMesh(m.vertices.tolist(), m.faces.tolist())

pv_rbc = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh)
ic_rbc = mir.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

box_lo = (          - domain[0],           - domain[1],             rc)
box_hi = (domain[0] + domain[0], domain[1] + domain[1], domain[2] - rc)
plates = mir.Walls.Box("plates", box_lo, box_hi, inside=True)
u.registerWall(plates, 0)

# fake repulsion module to force sdf computation
u.registerPlugins(mir.Plugins.createWallRepulsion("wallRepulsion", pv_rbc, plates, C=500, h=rc, max_force=500))

dump_every = 1
u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh('partDump', pv_rbc, dump_every, ["sdf"], 'h5/rbc-'))

u.run(2, dt=0)

# nTEST: dump.h5.mesh.sdf
# cd dump
# rm -rf h5
# mesh="../../data/rbc_mesh.off"
# mir.run --runargs "-n 2" ./h5.mesh.sdf.py --mesh $mesh
# mir.post h5dump -d sdf h5/rbc-00001.h5 | awk '{print $2}' > h5.mesh.sdf.out.txt
