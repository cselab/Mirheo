#!/usr/bin/env python

import numpy as np
import ymero as ymr

ranks  = (1, 1, 1)
domain = [4., 4., 4.]
density = 8

u = ymr.ymero(ranks, tuple(domain), dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.UniformSphere(density=density, center=(2., 2., 2.), radius=2.0, inside=True)
u.registerParticleVector(pv=pv, ic=ic)

u.run(2)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.uniform.sphere
# cd ic
# rm -rf pos*.txt vel*.txt
# ymr.run --runargs "-n 2" ./sphere.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
