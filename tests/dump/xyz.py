#!/usr/bin/env python

import udevicex as ymr

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = ymr.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

xyz = ymr.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
u.registerPlugins(xyz)

u.run(2)

# TEST: dump.xyz
# cd dump
# rm -rf xyz
# ymr.run --runargs "-n 2" ./xyz.py > /dev/null
# tail -n +3 xyz/pv_00000.xyz | sort > xyz.out.txt

