#!/usr/bin/env python

import mirheo as mir

dt = 0.001
density = 10

ranks  = (1, 1, 1)
domain = (16, 16, 16)
a = 1

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv1 = mir.ParticleVectors.ParticleVector('pv1', mass = 1)
pv2 = mir.ParticleVectors.ParticleVector('pv2', mass = 1)
ic = mir.InitialConditions.Uniform(density=density/2)
u.registerParticleVector(pv1, ic)
u.registerParticleVector(pv2, ic)
    
dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv1, pv2], sample_every, dump_every, bin_size, [("velocity", "vector_from_float4")], 'h5/solvent-'))

u.run(5002)

# nTEST: flow.double_poiseuille.2pvs
# cd flow
# rm -rf h5
# mir.run --runargs "-n 2" ./double_poiseuille.2pvs.py
# mir.avgh5 xz velocity h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt
