#!/usr/bin/env python

import sys
import numpy as np

import udevicex as ymr

import sys, argparse
sys.path.append("..")
from common.membrane_params import set_lina

parser = argparse.ArgumentParser()
parser.add_argument('--substep', dest='substep', action='store_true')
parser.set_defaults(substep=False)
args = parser.parse_args()

tend = 10.0
dt = 0.001
a = 1.0

if args.substep:
    substeps = 10
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv_flu = ymr.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_flu = ymr.InitialConditions.Uniform(density=8)
u.registerParticleVector(pv=pv_flu, ic=ic_flu)


mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)

ic_pos_rot = [[2.0, 5.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
              [4.0, 4.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
              [6.0, 3.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

ic_rbc   = ymr.InitialConditions.Membrane(ic_pos_rot)
u.registerParticleVector(pv_rbc, ic_rbc)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.01, dt=dt, power=0.25)
cnt = ymr.Interactions.LJ('cnt', 1.0, epsilon=0.4, sigma=0.35, max_force=400.0, object_aware=False)

prm_rbc = ymr.Interactions.MembraneParameters()

if prm_rbc:
    set_lina(1.0, prm_rbc)
    prm_rbc.rnd = False
    prm_rbc.dt = dt
    
int_rbc = ymr.Interactions.MembraneForces("int_rbc", prm_rbc, stressFree=True)

u.registerInteraction(dpd)
u.registerInteraction(cnt)

if args.substep:
    integrator = ymr.Integrators.SubStepMembrane('substep_membrane', dt, substeps, int_rbc)
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pv_rbc)
else:
    vv = ymr.Integrators.VelocityVerlet('vv', dt)
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)

u.setInteraction(dpd, pv_flu, pv_flu)
u.setInteraction(dpd, pv_flu, pv_rbc)
u.setInteraction(cnt, pv_rbc, pv_rbc)


vv_dp = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv_dp', dt=dt, force=a, direction='x')
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv_flu)


dump_mesh = ymr.Plugins.createDumpMesh("mesh_dump", pv_rbc, (int)(0.15/dt), "ply/")
u.registerPlugins(dump_mesh)


nsteps = (int) (tend/dt)
u.run(nsteps)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: contact.membrane.dp
# cd contact
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./membranes.dp.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: contact.membrane.dp.substep
# cd contact
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./membranes.dp.py --substep > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
