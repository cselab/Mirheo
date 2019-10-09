#!/usr/bin/env python

import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = [4., 6., 8.]

u = mir.Mirheo(ranks, tuple(domain), dt=0, debug_level=3, log_filename='log', no_splash=True)

a=(0.1, 0.2, 0.3)

coords = [[-a[0], -a[1], -a[2]],
          [-a[0], -a[1],  a[2]],
          [-a[0],  a[1], -a[2]],
          [-a[0],  a[1],  a[2]],
          [ a[0],  a[1], -a[2]],
          [ a[0],  a[1],  a[2]]]

com_q = [[ 1., 0., 0.,    1.0, 0.0, 0.0, 0.0],
         [ 3., 0., 0.,    1.0, 2.0, 0.0, 0.0],
         [-1., 0., 0.,    1.0, 0.0, 3.0, 0.0], # out of the domain
         [ 2., 0., 0.,    1.0, 0.0, 0.0, 1.0]]

pv = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=a)
ic = mir.InitialConditions.Rigid(com_q, coords)
u.registerParticleVector(pv, ic)

u.run(2)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.rigid
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./rigid.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
