#!/usr/bin/env python

import mirheo as mir

ranks  = (1, 1, 1)
domain = (8, 8, 16)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

axes = (1.0, 2.0, 3.0)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

ov = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic = mir.InitialConditions.Rigid(com_q=com_q, coords=coords)
u.registerParticleVector(ov, ic)

u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov, dump_every=1, filename="stats/ellipsoid.csv"))

u.run(2, dt=0)

# TEST: dump.rov_stats
# cd dump
# rm -rf stats rov_stats.out.txt
# mir.run --runargs "-n 2" ./rov_stats.py
# cp stats/ellipsoid.csv rov_stats.out.txt
