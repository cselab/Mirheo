#!/usr/bin/env python

import numpy as np
import ymero as ymr
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kb', type=float, default=0.0)
parser.add_argument('--C0', type=float, default=0.0)
parser.add_argument('--kad', type=float, default=0.0)
parser.add_argument('--DA0', type=float, default=0.0)
parser.add_argument('--ncells', type=int, default=1)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]]*args.ncells)
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc         = ymr.Interactions.MembraneParameters()
prm_bending_rbc = ymr.Interactions.JuelicherBendingParameters()

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

if prm_bending_rbc:
    prm_bending_rbc.kb   = args.kb
    prm_bending_rbc.C0   = args.C0
    prm_bending_rbc.kad  = args.kad
    prm_bending_rbc.DA0  = args.DA0
    
int_rbc = ymr.Interactions.MembraneForcesJuelicher("int_rbc", prm_rbc, prm_bending_rbc, stressFree=False)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_every = 1

u.registerPlugins(ymr.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["areas", "scalar"],
                                                           ["meanCurvatures", "scalar"],
                                                           ["forces", "vector"]],
                                                          "h5/rbc-"))

u.run(2)

# nTEST: membrane.bending.juelicher
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./juelicher.py --kb 1000.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00000.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.C0
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./juelicher.py --kb 1000.0 --C0 0.5 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00000.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.AD
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./juelicher.py --kad 1000.0 --DA0 1.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00000.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.multiple
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./juelicher.py --kb 1000.0 --C0 1.0 --kad 1000.0 --DA0 1.0 --ncells 4 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00000.h5 --out forces.out.txt
