#!/usr/bin/env python

import sys, trimesh, argparse
import numpy as np
import mirheo as mir
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', type=str)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh = trimesh.load_mesh(args.mesh)

mesh_rbc = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc)

vv = mir.Integrators.VelocityVerlet('vv')
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

u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraRbcForce", pv_rbc, forces.tolist()))
u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, 500, "ply/"))

u.run(5000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: membrane.extra_force
# cd membrane
# mesh="../../data/rbc_mesh.off"
# mir.run --runargs "-n 2" ./extra_force.py --mesh $mesh
# mv pos.rbc.txt pos.rbc.out.txt 
