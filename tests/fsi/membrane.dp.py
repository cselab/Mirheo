#!/usr/bin/env python

import sys
import numpy as np

import udevicex as udx

import sys
sys.path.append("..")
from common.membrane_params import set_lina

dt = 0.001
a = 1.0

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv_flu = udx.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_flu = udx.InitialConditions.Uniform(density=8)
u.registerParticleVector(pv=pv_flu, ic=ic_flu)


mesh_rbc = udx.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = udx.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = udx.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.01, dt=dt, power=0.25)

prm_rbc = udx.Interactions.MembraneParameters()

if prm_rbc:
    set_lina(1.0, prm_rbc)
    prm_rbc.rnd = False
    prm_rbc.dt = dt
    
int_rbc = udx.Interactions.MembraneForces("int_rbc", prm_rbc, stressFree=True)

u.registerInteraction(dpd)
u.registerInteraction(int_rbc)

u.setInteraction(int_rbc, pv_rbc, pv_rbc)
u.setInteraction(dpd, pv_flu, pv_flu)
u.setInteraction(dpd, pv_flu, pv_rbc)


vv    = udx.Integrators.VelocityVerlet('vv', dt)
vv_dp = udx.Integrators.VelocityVerlet_withPeriodicForce('vv_dp', dt=dt, force=a, direction='x')
u.registerIntegrator(vv)
u.registerIntegrator(vv_dp)

u.setIntegrator(vv,    pv_rbc)
u.setIntegrator(vv_dp, pv_flu)


# dump_mesh = udx.Plugins.createDumpMesh("mesh_dump", pv_rbc, 150, "ply/")
# u.registerPlugins(dump_mesh)

u.run(3000)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: fsi.membrane.dp
# cd fsi
# rm -rf pos.rbc.out.txt pos.rbc.txt
# cp ../../data/rbc_mesh.off .
# udx.run --runargs "-n 2" ./membrane.dp.py > /dev/null
# mv pos.rbc.txt pos.rbc.out.txt 
