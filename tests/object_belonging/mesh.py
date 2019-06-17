#!/usr/bin/env python

import ymero as ymr
import numpy as np

density = 4
path   = "ply/"
pvname = "rbc"
off    = "rbc_mesh.off"

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

mesh = ymr.ParticleVectors.MembraneMesh(off)

pv_rbc = ymr.ParticleVectors.MembraneVector(pvname, mass=1.0, mesh=mesh)
ic_rbc = ymr.InitialConditions.Membrane([[6.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

pv_outer = ymr.ParticleVectors.ParticleVector('pv_outer', mass = 1.0)
ic_outer = ymr.InitialConditions.Uniform(density)
u.registerParticleVector(pv_outer, ic_outer)

inner_checker = ymr.BelongingCheckers.Mesh("inner_solvent_checker")
u.registerObjectBelongingChecker(inner_checker, pv_rbc)

pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every = 0, inside = "pv_inner")

u.run(1)

if u.isMasterTask():
    pv_inner_pos = pv_inner.getCoordinates()
    np.savetxt("pos.inner.txt", pv_inner_pos)

# TEST: object_belonging.mesh
# cd object_belonging
# rm -rf ply/ pos.inner.txt 
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./mesh.py
# cat pos.inner.txt | LC_ALL=en_US.utf8 sort > belonging.out.txt
