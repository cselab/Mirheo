#!/usr/bin/env python

import sys, trimesh, argparse
import numpy as np
import ymero as ymr
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', type=str)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

mesh = trimesh.load_mesh(args.mesh)

mesh_rbc = ymr.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

forces = np.zeros((len(mesh.vertices), 3))
id_min = np.argmin(mesh.vertices[:,0])
id_max = np.argmax(mesh.vertices[:,0])
force_magn = 500.0
forces[id_min][0] = - force_magn
forces[id_max][0] = + force_magn

extraForces = ymr.Plugins.createMembraneExtraForce("extraRbcForce", pv_rbc, forces.tolist())
u.registerPlugins(extraForces)

dump_mesh = ymr.Plugins.createDumpMesh("mesh_dump", pv_rbc, 500, "ply/")
u.registerPlugins(dump_mesh)

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.extraForce
# cd membrane
# mesh="../../data/rbc_mesh.off"
# ymr.run --runargs "-n 2" ./extraForce.py --mesh $mesh > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
