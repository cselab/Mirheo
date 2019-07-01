#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = {
    "x0"     : 0.457,
    "ka_tot" : 0.0,
    "kv_tot" : 0.0,
    "ka"     : 0.0,
    "ks"     : 0.0,
    "mpow"   : 2,
    "gammaC" : 0.0,
    "gammaT" : 0.0,
    "kBT"    : 0.0,
    "tot_area"   : 62.2242,
    "tot_volume" : 26.6649,
    "kb"     : 1000.0,
    "theta"  : 0.0
}
    
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=False)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)


dump_every = 1

u.registerPlugins(mir.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["forces", "vector"]],
                                                          "h5/rbc-"))

u.run(2)

# nTEST: membrane.bending.kantor
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./kantor.py
# mir.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt
