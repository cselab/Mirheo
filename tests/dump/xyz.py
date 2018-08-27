#!/usr/bin/env python

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

xyz = udx.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
u.registerPlugins(xyz)

u.run(2)

# TEST: dump.xyz
# cd dump
# rm -rf xyz
# udx.run --runargs "-n 2" ./xyz.py > /dev/null
# cat xyz/pv_00000.xyz | sort > xyz.out.txt

