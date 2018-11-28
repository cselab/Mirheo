#!/usr/bin/env python

import numpy as np
import ymero as ymr
import sys, argparse

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, debug_level=3, log_filename='log')

mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc         = ymr.Interactions.MembraneParameters()
prm_bending_rbc = ymr.Interactions.KantorBendingParameters()

if prm_rbc:
    lscale = 1.0
    p              = 0.000906667 * lscale
    prm_rbc.x0        = 0.457    
    prm_rbc.ka        = 0.0
    prm_rbc.kd        = 0.0
    prm_rbc.kv        = 0.0
    prm_rbc.gammaC    = 0.0
    prm_rbc.gammaT    = 0.0
    prm_rbc.kbT       = 0.0
    prm_rbc.mpow      = 2.0
    prm_rbc.totArea   = 62.2242 * lscale**2
    prm_rbc.totVolume = 26.6649 * lscale**3

    prm_rbc.ks        = 0
    prm_rbc.rnd = False
    prm_rbc.dt = dt

if prm_bending_rbc:
    prm_bending_rbc.kb  = 1.0
    prm_bending_rbc.theta  = 0.0
    
int_rbc = ymr.Interactions.MembraneForcesKantor("int_rbc", prm_rbc, prm_bending_rbc, stressFree=False)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)


dump_every = 1

u.registerPlugins(ymr.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["forces", "vector"]],
                                                          "h5/rbc-"))

u.run(2)

# nTEST: membrane.bending.kantor
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./kantor.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
