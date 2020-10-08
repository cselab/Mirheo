#!/usr/bin/env python

import numpy as np
import mirheo as mir
import sys, argparse

sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--stress_free', action='store_true', default=False)
parser.add_argument('--sphere_stress_free', action='store_true', default=False)
parser.add_argument('--fluctuations', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")

inactive_id = 0
active_id = 1

pv_rbc = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc = mir.InitialConditions.MembraneWithTypeId([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0],
                                                   [4.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]],
                                                  [active_id, inactive_id])
u.registerParticleVector(pv_rbc, ic_rbc)

# only the active membrane gets forces
prm_rbc = lina_parameters(1.0, args.fluctuations)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stress_free,
                                          filter_desc="by_type_id", type_id=active_id)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh("mesh_dump", pv_rbc, 150, ["membrane_type_id"], "h5/mesh-"))

u.run(5000, dt=dt)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.filtered
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.filtered.py
# mv pos.rbc.txt pos.rbc.out.txt 
