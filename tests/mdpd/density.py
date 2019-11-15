#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (9, 9, 9)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=10)
u.registerParticleVector(pv, ic)

rc = 1.0
rd = 0.75

den  = mir.Interactions.Pairwise('den', rd, kind="Density", density_kernel="MDPD")
mdpd = mir.Interactions.Pairwise('mdpd', rc, kind="MDPD", rd=rd, a=10.0, b=20.0, gamma=10.0, kBT=0.1, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)


density_channel = "rho"
dump_every = 1000

u.registerPlugins(mir.Plugins.createParticleChannelSaver('rhoSaver', pv, "densities", density_channel))

grid_sample_every = 2
grid_dump_every   = 1000
grid_bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], grid_sample_every, grid_dump_every, grid_bin_size, [density_channel], 'h5/solvent-'))

u.run(5002)

# nTEST: mdpd.density
# cd mdpd
# rm -rf profile.out.txt h5
# mir.run --runargs "-n 2" ./density.py
# mir.avgh5 yz rho h5/solvent-000*.h5 > profile.out.txt
