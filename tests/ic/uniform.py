#!/usr/bin/env python

import numpy as np
import udevicex as udx

ranks  = (1, 1, 1)
domain = [4., 2., 3.]
density = 8

u = udx.udevicex(ranks, tuple(domain), debug_level=2, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)

# xyz = udx.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
# u.registerPlugins(xyz)

u.run(2)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.uniform
# cd ic
# rm -rf pos*.txt vel*.txt
# udx.run --runargs "-n 2" ./uniform.py > /dev/null
# paste pos.ic.txt vel.ic.txt | sort > ic.out.txt
