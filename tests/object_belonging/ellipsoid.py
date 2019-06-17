#!/usr/bin/env python

import ymero as ymr
import numpy as np

density = 4

ranks  = (1, 1, 1)

axes = (1.0, 2.0, 3.0)
fact = 3
domain = (fact*axes[0], fact*axes[1], fact*axes[2])

u = ymr.ymero(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

pv_ell = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = ymr.InitialConditions.Rigid(com_q, coords)
u.registerParticleVector(pv_ell, ic_ell)

pv_outer = ymr.ParticleVectors.ParticleVector('pv_outer', mass = 1.0)
ic_outer = ymr.InitialConditions.Uniform(density)
u.registerParticleVector(pv_outer, ic_outer)

inner_checker = ymr.BelongingCheckers.Ellipsoid("inner_solvent_checker")
u.registerObjectBelongingChecker(inner_checker, pv_ell)

pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every = 0, inside = "pv_inner")

u.run(1)

if u.isMasterTask():
    pv_inner_pos = pv_inner.getCoordinates()
    np.savetxt("pos.inner.txt", pv_inner_pos)

# TEST: object_belonging.ellipsoid
# cd object_belonging
# rm -rf ply/ pos.inner.txt 
# ymr.run --runargs "-n 2" ./ellipsoid.py
# cat pos.inner.txt | LC_ALL=en_US.utf8 sort > belonging.out.txt
