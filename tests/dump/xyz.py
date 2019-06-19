#!/usr/bin/env python

import ymero as ymr

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = ymr.ymero(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv, ic)

u.registerPlugins(ymr.Plugins.createDumpXYZ('xyz', pv, 1, "xyz"))

u.run(2)

# TEST: dump.xyz
# cd dump
# rm -rf xyz
# ymr.run --runargs "-n 2" ./xyz.py
# tail -n +3 xyz/pv_00000.xyz | LC_ALL=en_US.utf8 sort > xyz.out.txt

