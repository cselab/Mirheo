#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

rc = 1.0
num_density = 10

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.UniformWithPolChain(num_density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc, kind='ViscoElasticDPD',
                                a=10.0, gamma=10.0, kBT=1.0, power=0.5,
                                H=1.0, friction=1.0, kBTC=1.0, n0=num_density)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerletPolChain('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', every=1000, filename="stats.csv"))

u.run(5001, dt=dt)

# nTEST: vedpd.rest
# cd vedpd
# rm -rf stats.csv
# mir.run --runargs "-n 2" ./rest.py > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt
