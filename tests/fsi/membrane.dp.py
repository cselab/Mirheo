#!/usr/bin/env python

import numpy as np
import ymero as ymr

import sys, argparse
sys.path.append("..")
from common.membrane_params import set_lina
from common.membrane_params import set_lina_bending

parser = argparse.ArgumentParser()
parser.add_argument('--substep', dest='substep', action='store_true')
parser.set_defaults(substep=False)
args = parser.parse_args()

tend = 3.0
dt = 0.001
a = 1.0

if args.substep:
    substeps = 10
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv_flu = ymr.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_flu = ymr.InitialConditions.Uniform(density=8)
u.registerParticleVector(pv=pv_flu, ic=ic_flu)


mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.01, power=0.25)

prm_rbc         = ymr.Interactions.MembraneParameters()
prm_bending_rbc = ymr.Interactions.KantorBendingParameters()

if prm_rbc:
    set_lina(1.0, prm_rbc)
    prm_rbc.rnd = False
if prm_bending_rbc:
    set_lina_bending(1.0, prm_bending_rbc)
    
int_rbc = ymr.Interactions.MembraneForcesKantor("int_rbc", prm_rbc, prm_bending_rbc, stressFree=True)

u.registerInteraction(dpd)

if args.substep:
    integrator = ymr.Integrators.SubStepMembrane('substep_membrane', substeps, int_rbc)
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pv_rbc)
else:
    vv = ymr.Integrators.VelocityVerlet('vv')
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)

u.setInteraction(dpd, pv_flu, pv_flu)
u.setInteraction(dpd, pv_flu, pv_rbc)



vv_dp = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv_dp', force=a, direction='x')
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv_flu)


# dump_mesh = ymr.Plugins.createDumpMesh("mesh_dump", pv_rbc, (int)(0.15/dt), "ply/")
# u.registerPlugins(dump_mesh)


nsteps = (int) (tend/dt)
u.run(nsteps)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: fsi.membrane.dp
# cd fsi
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./membrane.dp.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: fsi.membrane.dp.substep
# cd fsi
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./membrane.dp.py --substep > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
