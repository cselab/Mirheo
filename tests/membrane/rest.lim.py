#!/usr/bin/env python

import numpy as np
import ymero as ymr
import sys, argparse

sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--stressFree', action='store_true', default=False)
parser.add_argument('--sphereStressFree', action='store_true', default=False)
parser.add_argument('--fluctuations', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

if args.sphereStressFree:
    mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off", "sphere_mesh.off")
else:
    mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")

pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=10.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0, args.fluctuations)

prm_rbc["ka"] = 200.0
prm_rbc["mu"] = 100.0
prm_rbc["a3"] = 0.0
prm_rbc["a4"] = 0.0
prm_rbc["b1"] = 0.0
prm_rbc["b2"] = 0.0

int_rbc = ymr.Interactions.MembraneForces("int_rbc", "Lim", "Kantor", **prm_rbc, stress_free=True)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_every = 100

u.registerPlugins(ymr.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["forces", "vector"]],
                                                          "h5/rbc-"))
u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest.Lim
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.lim.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

