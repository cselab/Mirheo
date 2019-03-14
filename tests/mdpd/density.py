#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (9, 9, 9)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0
rd = 0.75

den  = ymr.Interactions.Density('den', rd, kernel="MDPD")
mdpd = ymr.Interactions.MDPD('mdpd', rc, rd, a=10.0, b=20.0, gamma=10.0, kbt=0.1, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)


density_channel = "rho"
dump_every = 1000

#u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", dump_every))
u.registerPlugins(ymr.Plugins.createParticleChannelSaver('rhoSaver', pv, "densities", density_channel))
#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, dump_every, [[density_channel, "scalar"]], 'h5/solvent_particles-'))

gridSampleEvery = 2
gridDumpEvery   = 1000
gridBinSize     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], gridSampleEvery, gridDumpEvery, gridBinSize, [(density_channel, "scalar")], 'h5/solvent-'))

u.run(5002)

# nTEST: mdpd.density
# cd mdpd
# rm -rf profile.out.txt h5
# ymr.run --runargs "-n 2" ./density.py > /dev/null
# ymr.avgh5 yz rho h5/solvent-000*.h5 > profile.out.txt
