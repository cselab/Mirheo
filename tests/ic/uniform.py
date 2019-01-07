#!/usr/bin/env python

import numpy as np
import ymero as ymr

ranks  = (1, 1, 1)
domain = [4., 2., 3.]
density = 8

u = ymr.ymero(ranks, tuple(domain), dt=0, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=density)
u.registerParticleVector(pv=pv, ic=ic)

# xyz = ymr.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
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
# ymr.run --runargs "-n 2" ./uniform.py > /dev/null
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
