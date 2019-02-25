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

pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0, args.fluctuations)    
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stressFree)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

u.registerPlugins(ymr.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.rest
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.py --stressFree > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stressFree.sphere
# cd membrane
# cp ../../data/rbc_mesh.off .
# cp ../../data/sphere_mesh.off .
# ymr.run --runargs "-n 2" ./rest.py --stressFree --sphereStressFree > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.py --fluctuations > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: membrane.rest.stressfree.fluctuations
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./rest.py --stressFree --fluctuations > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
