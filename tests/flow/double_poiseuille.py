#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)
a = 1

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv, ic)
    
dpd = mir.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run(5002)

# nTEST: flow.double_poiseuille
# cd flow
# rm -rf h5
# mir.run --runargs "-n 2" ./double_poiseuille.py
# mir.avgh5 xz velocity h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt
