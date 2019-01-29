#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0
rd = 0.75

den  = ymr.Interactions.Density('den', rd)
mdpd = ymr.Interactions.MDPD('mdpd', rc, rd, a=10.0, b=-5.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", 1000))
#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, 1000, [['densities', 'scalar']], 'h5/solvent_particles-'))

u.run(5001)

# nTEST: mdpd.rest
# cd mdpd
# rm -rf stats.txt
# ymr.run --runargs "-n 2" ./rest.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

