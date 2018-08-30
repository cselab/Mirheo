#!/usr/bin/env python

import udevicex as udx

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

xyz = udx.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
u.registerPlugins(xyz)

u.run(2)

# nTEST: dump.xyz
# cd dump
# rm -rf xyz
# udx.run --runargs "-n 2" ./xyz.py > /dev/null
# tail -n +3 xyz/pv_00000.xyz | sort > xyz.out.txt

