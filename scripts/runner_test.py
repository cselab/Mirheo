#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:44:15 2018

@author: alexeedm
"""

import uDeviceX as udx

dt = 0.001
rho = 8.00
#generate_ic(fname)
#....
sim = udx.Simulation( domain=(32, 32, 32), ranks=(1, 1, 2), debug_lvl=3 )

ic = udx.UniformIC(rho)
pv = udx.SimplePV(ic, mass=1.5)
sim.addParticleVector(pv)

sph_ic = udx.RigidIC()
sphere = udx.RigidEllipsoid(sph_ic, particles_per_object=10, semi_axes=(1, 1, 1))
sim.addParticleVector(sphere)

dpd = udx.DPD(1.0, a=10, gamma=5, kbt=1.0, dt=dt, power=0.5)
dpd.addParticleVectors(pv, pv, gamma=1.0)
sim.addInteraction(dpd)


integrator = udx.VelocityVerlet(dt)
integrator.addParticleVector(pv)
sim.addIntegrator(integrator)

i2 = udx.Rigid_VelocityVerlet(dt)
i2.addParticleVector(sphere)
sim.addIntegrator(i2)

xml = sim.generate(10, './test.xml')

    
print(xml)