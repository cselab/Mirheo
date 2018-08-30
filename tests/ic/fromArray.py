#!/usr/bin/env python

import numpy as np
import udevicex as udx

ranks  = (1, 1, 1)
domain = [4., 6., 8.]

u = udx.udevicex(ranks, tuple(domain), debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)

pos = [[a*domain[0], a*domain[1], a*domain[2]] for a in [0.1, 0.5, 0.8, 1.5]] # one particle is outside
v=[1., 2., 3.]
vel = [[a*v[0], a*v[1], a*v[2]] for a in [0.1, 0.5, 0.8, 1.5]]

ic = udx.InitialConditions.FromArray(pos=pos, vel=vel)
u.registerParticleVector(pv=pv, ic=ic)

xyz = udx.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
u.registerPlugins(xyz)


u.run(2)

# nTEST: ic.fromArray
# cd ic
# rm -rf xyz/ xyz.out.txt
# udx.run --runargs "-n 2" ./fromArray.py > /dev/null
# tail -n +3 xyz/pv_00000.xyz | sort > xyz.out.txt
