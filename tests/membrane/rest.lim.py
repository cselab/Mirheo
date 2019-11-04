#!/usr/bin/env python

import numpy as np
import mirheo as mir
import sys

sys.path.append("..")
from common.membrane_params import lina_parameters

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")

pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=10.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)

prm_rbc["ka"] = 200.0
prm_rbc["mu"] = 100.0
prm_rbc["a3"] = 0.0
prm_rbc["a4"] = 0.0
prm_rbc["b1"] = 0.0
prm_rbc["b2"] = 0.0
prm_rbc.pop("ks")
prm_rbc.pop("mpow")
prm_rbc.pop("x0")
int_rbc = mir.Interactions.MembraneForces("int_rbc", "Lim", "Kantor", **prm_rbc, stress_free=True)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_every = 500

u.registerPlugins(mir.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          ["forces"],
                                                          "h5/rbc-"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.Lim
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.lim.py
# mv pos.rbc.txt pos.rbc.out.txt 

