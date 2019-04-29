#!/usr/bin/env python

import ymero as ymr

ranks  = (1, 1, 1)
domain = (8, 8, 16)

u = ymr.ymero(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

axes = (1.0, 2.0, 3.0)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

ov = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
u.registerParticleVector(ov, ic)

u.registerPlugins(ymr.Plugins.createDumpObjectStats("objStats", ov, dump_every=1, path="stats"))

u.run(2)

# TEST: dump.rigidStats
# cd dump
# rm -rf stats rigidStats.out.txt
# ymr.run --runargs "-n 2" ./rigidStats.py
# cp stats/ellipsoid.txt rigidStats.out.txt

