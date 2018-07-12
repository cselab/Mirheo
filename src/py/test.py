#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:35:18 2018

@author: alexeedm
"""

import sys
sys.path.append('/home/alexeedm/udevicex/build')
import _udevicex as udx
import numpy as np

dt = 0.001

u = udx.udevicex((1,1,1), (2, 2, 2), debug_level=10, log_filename='stdout')

pv = udx.ParticleVector('pv', 1)
ic = udx.UniformIC(density=2)
u.registerParticleVector(pv=pv, ic=ic)

dpd = udx.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.VelocityVerlet('vv', dt=dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

print(u)

u.run(5)

print(pv)

u.run(5)

coo = pv.getVelocities()

print(coo)

#print(vv)   