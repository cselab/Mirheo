#!/usr/bin/env python

import numpy as np
import mirheo as mir

import sys, argparse
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--stress_free', action='store_true', default=False)
args = parser.parse_args()

dt = 0.01
substeps = 10

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stress_free)

integrator = mir.Integrators.SubStep('substep_membrane', substeps, int_rbc)
u.registerIntegrator(integrator)
u.setIntegrator(integrator, pv_rbc)

# Note that the interaction is NOT registered inside `u`

u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.substep
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.substep.py
# mv pos.rbc.txt pos.rbc.out.txt 
