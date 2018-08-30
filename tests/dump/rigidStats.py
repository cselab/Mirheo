#!/usr/bin/env python

import udevicex as udx

ranks  = (1, 1, 1)
domain = (8, 8, 16)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

axes = (1.0, 2.0, 3.0)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

OV = udx.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic = udx.InitialConditions.Rigid(com_q=com_q, coords=coords)
u.registerParticleVector(pv=OV, ic=ic)


ovStats = udx.Plugins. createDumpObjectStats("objStats", ov=OV, dump_every=1, path="stats")
u.registerPlugins(ovStats)

u.run(2)

# TEST: dump.rigidStats
# cd dump
# rm -rf stats rigidStats.out.txt
# udx.run --runargs "-n 2" ./rigidStats.py > /dev/null
# cp stats/ellipsoid.txt rigidStats.out.txt

