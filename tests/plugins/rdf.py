#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

rc = 1.0
num_density = 8

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(num_density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc, kind='DPD', a=100.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createRdf('rdf', pv, max_dist=2*rc, nbins=50, basename="rdf/pv-", every=5000))

u.run(5002)

# nTEST: plugins.rdf
# cd plugins
# rm -rf rdf rdf.out.txt
# mir.run --runargs "-n 2" ./rdf.py
# mir.post ../tools/dump_csv.py rdf/pv-00001.csv r rdf > rdf.out.txt
