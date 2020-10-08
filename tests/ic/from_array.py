#!/usr/bin/env python

import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = [4., 6., 8.]

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)

pos = [[a*domain[0], a*domain[1], a*domain[2]] for a in [0.1, 0.5, 0.8, 1.5]] # one particle is outside
v=[1., 2., 3.]
vel = [[a*v[0], a*v[1], a*v[2]] for a in [0.1, 0.5, 0.8, 1.5]]

ic = mir.InitialConditions.FromArray(pos, vel)
u.registerParticleVector(pv, ic)

u.run(2, dt=0)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.fromArray
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./from_array.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
