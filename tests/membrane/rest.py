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

if args.sphere_stress_free:
    mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off", "sphere_mesh.off")
else:
    mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")

pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0, args.fluctuations)    
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stress_free)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/"))

u.run(5000, dt=dt)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.py
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stress_free
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.py --stress_free
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stress_free.sphere
# cd membrane
# cp ../../data/rbc_mesh.off .
# cp ../../data/sphere_mesh.off .
# mir.run --runargs "-n 2" ./rest.py --stress_free --sphere_stress_free
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.py --fluctuations
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stress_free.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./rest.py --stress_free --fluctuations
# mv pos.rbc.txt pos.rbc.out.txt 
