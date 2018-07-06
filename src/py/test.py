#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:35:18 2018

@author: alexeedm
"""

import sys
sys.path.append('/home/alexeedm/udevicex/build')
import _udevicex as udx

vv = udx.VelocityVerlet('haha', 2.0, (1.0, 2.0, 3))
pv = udx.SimpleParticleVector('pv', 1)
ic = udx.UniformIC(8)


u = udx.uDeviceX((1, 1, 1), (2, 2, 2), 'stdout', 10, False)

u.registerIntegrator(vv)
u.registerParticleVector(pv, ic, 0)

u.run(10)

print(vv)   