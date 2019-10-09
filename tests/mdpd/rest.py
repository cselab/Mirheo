#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density=3)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0
rd = 0.75

den  = mir.Interactions.Pairwise('den', rd, kind="Density", density_kernel="MDPD")
mdpd = mir.Interactions.Pairwise('mdpd', rc, kind="MDPD", rd=rd, a=10.0, b=20.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', "stats.txt", 1000))

u.run(5001)

# nTEST: mdpd.rest
# cd mdpd
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./rest.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

