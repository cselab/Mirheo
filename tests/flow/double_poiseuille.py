#!/usr/bin/env python3

import sys
sys.path.append('/home/amlucas/dimudx/build')
import _udevicex as udx
import numpy as np

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

if u.isComputeTask():
    pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
    ic = udx.InitialConditions.Uniform(density=4)
    u.registerParticleVector(pv=pv, ic=ic)
    
    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    
    vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

flag = u.isComputeTask()
stats = udx.Plugins.createStats('diagnostics', 100, flag)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

field = udx.Plugins.createDumpAveragePlugin('field', pv, sampleEvery, dumpEvery,
                                            binSize,
                                            [('velocity', 'vector')],
                                            './h5', flag)
u.registerPlugins(stats[0], stats[1])
u.registerPlugins(field[0], field[1])

u.run(2000)

# sTEST: double_poiseuille
# mpirun -n 2 ./double_poiseuille.py
# TODO
