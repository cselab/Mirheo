#!/usr/bin/env python

import mirheo as mir

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = mir.Mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=2)
u.registerParticleVector(pv, ic)

u.registerPlugins(mir.Plugins.createDumpXYZ('xyz', pv, 1, "xyz"))

u.run(2)

# TEST: dump.xyz
# cd dump
# rm -rf xyz
# mir.run --runargs "-n 2" ./xyz.py
# tail -n +3 xyz/pv_00000.xyz | LC_ALL=en_US.utf8 sort > xyz.out.txt

