#!/usr/bin/env python

import numpy as np
import mirheo as mir
import sys, argparse

sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--stress_free',  action='store_true', default=False)
parser.add_argument('--fluctuations', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")

pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0, args.fluctuations)
prm_rbc["kb"] = 44.0
prm_rbc["C0"] = 0.0
prm_rbc["kad"] = 0.0
prm_rbc["DA0"] = 0.0
prm_rbc.pop("theta")
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Juelicher", **prm_rbc, stress_free=args.stress_free)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_every = 150

u.registerPlugins(mir.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["areas", "scalar"],
                                                           ["meanCurvatures", "scalar"],
                                                           ["forces", "vector"]],
                                                          "h5/rbc-"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.juelicher
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.juelicher.py
# mv pos.rbc.txt pos.rbc.out.txt 
