#!/usr/bin/env python

import numpy as np
import ymero as ymr

import sys, argparse
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--stressFree', dest='stressFree', action='store_true')
parser.add_argument('--fluctuations', dest='rnd', action='store_true')
parser.set_defaults(stressFree=False)
parser.set_defaults(rnd=False)
args = parser.parse_args()

dt = 0.01
substeps = 10

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stressFree)

integrator = ymr.Integrators.SubStep('substep_membrane', substeps, int_rbc)
u.registerIntegrator(integrator)
u.setIntegrator(integrator, pv_rbc)

# Note that the interaction is NOT registered inside `u`


u.registerPlugins(ymr.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.substep
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.substep.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
