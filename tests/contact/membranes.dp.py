#!/usr/bin/env python

import numpy as np
import mirheo as mir
import sys, argparse
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--substep', action='store_true', default=False)
args = parser.parse_args()

tend = 10.0
dt = 0.001
a = 1.0

if args.substep:
    substeps = 10
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv_sol = mir.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_sol = mir.InitialConditions.Uniform(number_density=8)
u.registerParticleVector(pv=pv_sol, ic=ic_sol)


mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)

ic_pos_rot = [[2.0, 5.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
              [4.0, 4.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
              [6.0, 3.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

ic_rbc   = mir.InitialConditions.Membrane(ic_pos_rot)
u.registerParticleVector(pv_rbc, ic_rbc)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.001, power=0.5)
cnt = mir.Interactions.Pairwise('cnt', rc=1.0, kind="RepulsiveLJ", epsilon=0.14, sigma=0.4, max_force=400.0)

prm_rbc = lina_parameters(1.0)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=True)

u.registerInteraction(dpd)
u.registerInteraction(cnt)

if args.substep:
    integrator = mir.Integrators.SubStep('substep_membrane', substeps, [int_rbc])
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pv_rbc)
else:
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)

u.setInteraction(dpd, pv_sol, pv_sol)
u.setInteraction(dpd, pv_sol, pv_rbc)
u.setInteraction(cnt, pv_rbc, pv_rbc)


vv_dp = mir.Integrators.VelocityVerlet_withPeriodicForce('vv_dp', force=a, direction='x')
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv_sol)


u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, (int)(0.15/dt), "ply/"))


nsteps = (int) (tend/dt)
u.run(nsteps, dt=dt)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: contact.membrane.dp
# cd contact
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./membranes.dp.py
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: contact.membrane.dp.substep
# cd contact
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./membranes.dp.py --substep
# mv pos.rbc.txt pos.rbc.out.txt 
