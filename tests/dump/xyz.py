#!/usr/bin/env python3

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

if u.isComputeTask():
    pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
    ic = udx.InitialConditions.Uniform(density=2)
    u.registerParticleVector(pv=pv, ic=ic)
    
    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    
    vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

flag = u.isComputeTask()
xyz = udx.Plugins.createDumpXYZ('xyz', pv, 100, "xyz/", flag)
u.registerPlugins(xyz[0], xyz[1])

u.run(1)

# sTEST: dump.xyz
# mpirun -n 2 ./xyz.py

