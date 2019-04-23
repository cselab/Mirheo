#!/usr/bin/env python

import numpy as np
import ymero as ymr

ranks  = (1, 1, 1)
domain = [16, 16, 16]

u = ymr.ymero(ranks, tuple(domain), dt=0, debug_level=3, log_filename='log')

com_q = [[ 1., 0., 0.,    1.0, 0.0, 0.0, 0.0],
         [ 5., 0., 0.,    1.0, 2.0, 0.0, 0.0],
         [-9., 0., 0.,    1.0, 0.0, 3.0, 0.0], # out of the domain
         [ 0., 7., 0.,    1.0, 0.0, 0.0, 1.0]]

def center_line(s):
    L = 5.0
    P = 1.0
    R = 1.0
    t = s * L * np.pi / P 
    return (R * np.cos(t),
            R * np.sin(t),
            (s-0.5) * L)

def torsion(s):
    return 0.0

rv = ymr.ParticleVectors.RodVector('rod', mass=1, num_segments = 100)
ic = ymr.InitialConditions.Rod(com_q, center_line, torsion)
u.registerParticleVector(rv, ic)

dump_every = 1
u.registerPlugins(ymr.Plugins.createDumpParticles('rod_dump', rv, dump_every, [], 'h5/rod_particles-'))

u.run(2)

if rv:
    icpos = rv.getCoordinates()
    icvel = rv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    
del u

# TEST: ic.rod
# cd ic
# rm -rf pos*.txt vel*.txt
# ymr.run --runargs "-n 2" ./rod.py > /dev/null
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
