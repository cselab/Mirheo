#!/usr/bin/env python

import sys
import numpy as np

import udevicex as udx

import sys, argparse
sys.path.append("..")
from common.membrane_params import set_lina

parser = argparse.ArgumentParser()
parser.add_argument('--stressFree', dest='stressFree', action='store_true')
parser.add_argument('--fluctuations', dest='rnd', action='store_true')
parser.set_defaults(stressFree=False)
parser.set_defaults(rnd=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

mesh_rbc = udx.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = udx.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = udx.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = udx.Interactions.MembraneParameters()

if prm_rbc:
    set_lina(1.0, prm_rbc)
    prm_rbc.rnd = args.rnd
    prm_rbc.dt = dt
    
int_rbc = udx.Interactions.MembraneForces("int_rbc", prm_rbc, stressFree=args.stressFree)
vv = udx.Integrators.VelocityVerlet('vv', dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_mesh = udx.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/")
u.registerPlugins(dump_mesh)

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest
# cd membrane
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./rest.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./rest.py --stressFree > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./rest.py --fluctuations > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stressfree.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./rest.py --stressFree --fluctuations > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
