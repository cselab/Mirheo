#!/usr/bin/env python

import mirheo as mir

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = mir.Mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

mass = 1.0

mesh = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

ov  = mir.ParticleVectors.MembraneVector("rbc", mass, mesh)
ic = mir.InitialConditions.Membrane(com_q)

u.registerParticleVector(ov, ic)

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov, dump_every=1, path="stats"))

u.run(2)

# TEST: dump.ov_stats
# cd dump
# rm -rf stats ov_stats.out.txt
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./ov_stats.py
# cat stats/rbc.csv > ov_stats.out.txt
