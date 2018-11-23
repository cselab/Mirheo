#!/usr/bin/env python

import numpy as np
import udevicex as ymr

ranks  = (1, 1, 1)
domain = [4., 6., 8.]

u = ymr.udevicex(ranks, tuple(domain), debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)

pos = [[a*domain[0], a*domain[1], a*domain[2]] for a in [0.1, 0.5, 0.8, 1.5]] # one particle is outside
v=[1., 2., 3.]
vel = [[a*v[0], a*v[1], a*v[2]] for a in [0.1, 0.5, 0.8, 1.5]]

ic = ymr.InitialConditions.FromArray(pos=pos, vel=vel)
u.registerParticleVector(pv=pv, ic=ic)

# xyz = ymr.Plugins.createDumpXYZ('xyz', pv, 1, "xyz/")
# u.registerPlugins(xyz)

u.run(2)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.fromArray
# cd ic
# rm -rf pos*.txt vel*.txt
# ymr.run --runargs "-n 2" ./fromArray.py > /dev/null
# paste pos.ic.txt vel.ic.txt | sort > ic.out.txt
