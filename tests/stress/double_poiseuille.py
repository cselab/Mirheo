#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (32, 16, 16)
a = 1

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=4)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.01, power=0.5, stress=True, stress_period=sample_every*dt)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size,
                                                [("velocity", "vector_from_float4"),
                                                 ("stresses", "tensor6")],
                                                'h5/solvent-'))

u.run(5002)

# nTEST: stress.double_poiseuille
# cd stress
# rm -rf h5
# mir.run --runargs "-n 2" ./double_poiseuille.py
# mir.avgh5 xz stresses h5/solvent-0000[2-4].h5 | awk '{print $2}' > stress.out.txt
