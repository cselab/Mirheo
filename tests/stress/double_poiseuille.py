#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (32, 16, 16)
a = 1

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.01, power=0.5, stress=True, stress_period=sample_every*dt)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

field = ymr.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size,
                                      [("velocity", "vector_from_float8"),
                                       ("stresses", "tensor6")],
                                      'h5/solvent-')
u.registerPlugins(field)

u.run(5002)

# nTEST: stress.double_poiseuille
# cd stress
# rm -rf h5
# ymr.run --runargs "-n 2" ./double_poiseuille.py > /dev/null
# ymr.avgh5 xz stresses h5/solvent-0000[2-4].h5 | awk '{print $2}' > stress.out.txt
