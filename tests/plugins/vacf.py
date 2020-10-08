#!/usr/bin/env python

import mirheo as mir
import numpy as np

ranks  = (1, 1, 1)
domain = (32, 32, 32)

dt = 0.001

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density = 10)
u.registerParticleVector(pv=pv, ic=ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

t_eq = 2.0
t_measure = 0.2

u.registerPlugins(mir.Plugins.createVacf('vacf', pv, t_eq, t_eq + t_measure, dump_every=1, path='vacf'))

t_tot = t_eq + t_measure

u.run(int(t_tot / dt) + 1, dt=dt)

# nTEST: plugins.vacf
# cd plugins
# rm -rf vacf vacf.out.txt
# mir.run --runargs "-n 2" ./vacf.py
# mir.post ../tools/dump_csv.py vacf/pv.csv vacf > vacf.out.txt
