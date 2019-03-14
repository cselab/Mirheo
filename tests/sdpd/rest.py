#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

u = ymr.ymero(ranks, domain, dt, debug_level=8, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0

den  = ymr.Interactions.Density('den', rc, kernel="WendlandC2")
sdpd = ymr.Interactions.SDPD('sdpd', rc, viscosity=10.0, kBT=1.0, EOS="Linear", sound_speed=10.0, density_kernel="WendlandC2")
u.registerInteraction(den)
u.registerInteraction(sdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(sdpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", 1000))
#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, 1000, [], 'h5/solvent_particles-'))

u.run(5001)

# nTEST: sdpd.rest
# cd sdpd
# rm -rf stats.txt
# ymr.run --runargs "-n 2" ./rest.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

