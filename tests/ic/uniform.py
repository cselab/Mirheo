#!/usr/bin/env python

import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = [4., 2., 3.]
density = 8

u = mir.Mirheo(ranks, tuple(domain), dt=0, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

u.run(2)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.uniform
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./uniform.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
