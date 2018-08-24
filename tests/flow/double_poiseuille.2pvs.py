#!/usr/bin/env python

import udevicex as udx

dt = 0.001
density = 10

ranks  = (1, 1, 1)
domain = (16, 16, 16)
a = 1

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

pv1 = udx.ParticleVectors.ParticleVector('pv1', mass = 1)
pv2 = udx.ParticleVectors.ParticleVector('pv2', mass = 1)
ic = udx.InitialConditions.Uniform(density=density/2)
u.registerParticleVector(pv=pv1, ic=ic)
u.registerParticleVector(pv=pv2, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)

vv = udx.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=dt, force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

field = udx.Plugins.createDumpAverage('field', [pv1, pv2], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)

u.run(5002)

# nTEST: double_poiseuille.2pvs
# cd flow
# rm -rf h5
# udx.run -n 2 ./double_poiseuille.2pvs.py > /dev/null
# udx.avgh5 xz velocity h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt
