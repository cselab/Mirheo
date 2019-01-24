#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)
a = 1

sampleEvery = 2
dumpEvery   = 1000
binSize     = (1., 1., 1.)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPDWithStress('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5, stressPeriod=sampleEvery*dt)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize,
                                      [("velocity", "vector_from_float8"), ("stresses", "tensor6")], 'h5/solvent-')
u.registerPlugins(field)

u.run(5002)

# nTEST: stress.double_poiseuille
# cd stress
# rm -rf h5
# ymr.run --runargs "-n 2" ./double_poiseuille.py > /dev/null
# ymr.avgh5 xz stresses h5/solvent-0000[2-4].h5 | awk '{print $2}' > stress.out.txt
