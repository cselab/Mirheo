#!/usr/bin/env python

import mirheo as mir
import numpy as np

density = 4
ranks  = (1, 1, 1)

axes = (1.0, 2.0, 3.0)
fact = 3
domain = (fact*axes[0], fact*axes[1], fact*axes[2])

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

coords = [[-axes[0], -axes[1], -axes[2]],
          [ axes[0],  axes[1],  axes[2]]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = mir.InitialConditions.Rigid(com_q, coords)
u.registerParticleVector(pv_ell, ic_ell)

pv_outer = mir.ParticleVectors.ParticleVector('pv_outer', mass = 1.0)
ic_outer = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv_outer, ic_outer)

inner_checker = mir.BelongingCheckers.Ellipsoid("inner_solvent_checker")
u.registerObjectBelongingChecker(inner_checker, pv_ell)

pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every = 0, inside = "pv_inner")

u.run(1, dt=0)

if u.isMasterTask():
    pv_inner_pos = pv_inner.getCoordinates()
    np.savetxt("pos.inner.txt", pv_inner_pos)

# TEST: object_belonging.ellipsoid
# cd object_belonging
# rm -rf pos.inner.txt belonging.out.txt
# mir.run --runargs "-n 1" ./ellipsoid.py
# cat pos.inner.txt | LC_ALL=en_US.utf8 sort > belonging.out.txt
