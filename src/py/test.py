#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:35:18 2018

@author: alexeedm
"""

import sys
#sys.path.append('/home/amlucas/dimudx/build')
import udevicex as udx
import numpy as np

dt = 0.001

u = udx.initialize((1,1,1), (32, 32, 32), debug_level=3, log_filename='stdout')
r = udx.ParticleVectors.ParticleVector('pv', 1)
ic = udx.InitialConditions.Uniform(8)
integr = udx.Integrators.VelocityVerlet('vv', dt)
dpd = udx.Interactions.DPD('dpd', a=10, gamma=10, dt=dt, kbt=1.0, rc=1, power=0.5)

stats = udx.Plugins.createStats('st', 2)
v = udx.Plugins.createAddForce('f', r, (0,0,1))

#im = udx.Plugins.createImposeVelocity('nnn', r, 100, (0,0,0), (10,10,10), (1,1,1))

u.registerParticleVector(r, ic)
u.registerPlugins(stats[0], stats[1])
u.registerPlugins(v[0], v[1])
u.registerIntegrator(integr)
u.setIntegrator(integr, r)

u.registerInteraction(dpd)
u.setInteraction(dpd, r, r)
#u.registerPlugins(v[0], v[1])

u.run(10)
