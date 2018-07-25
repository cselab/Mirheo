#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")
from common.context import udevicex as udx
from common.membrane_params import lina as params_lina

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = udx.udevicex(ranks, domain, debug_level=8, log_filename='log')

mesh = udx.ParticleVectors.MembraneMesh("rbc_mesh.off")
rbc  = udx.ParticleVectors.MembraneVector("rbc", mass=1.0, object_size=498, mesh=mesh)
icrbc = udx.InitialConditions.Membrane("rbcs-ic.txt")
u.registerParticleVector(pv=rbc, ic=icrbc)

mdump = udx.Plugins.createDumpMesh("mesh_dump", rbc, 1, "ply/")
u.registerPlugins(mdump)

u.run(3)

# nTEST: dump.mesh
# cd dump
# echo "6.0 4.0 5.0 1.0 0.0 0.0 0.0" > rbcs-ic.txt
# cp ../../data/rbc_mesh.off .
# udx.run -n 2 ./mesh.py > /dev/null
# ply2punto ply/rbc_00000.ply > ply.out.txt
