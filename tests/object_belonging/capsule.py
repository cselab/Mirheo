#!/usr/bin/env python

import mirheo as mir
import numpy as np

density = 4
ranks  = (1, 1, 1)

R = 2.5
L = 3.0

fact = 3
domain = (fact*R, fact*R, fact*(L/2+R))

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

coords = [[-R, -R, -L/2-R],
          [ R,  R,  L/2+R]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

pv_cap = mir.ParticleVectors.RigidCapsuleVector('capsule', mass=1, object_size=len(coords), radius=R, length=L)
ic_cap = mir.InitialConditions.Rigid(com_q, coords)
u.registerParticleVector(pv_cap, ic_cap)

pv_outer = mir.ParticleVectors.ParticleVector('pv_outer', mass = 1.0)
ic_outer = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv_outer, ic_outer)

inner_checker = mir.BelongingCheckers.Capsule("inner_solvent_checker")
u.registerObjectBelongingChecker(inner_checker, pv_cap)

pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every = 0, inside = "pv_inner")

u.run(1, dt=0)

if u.isMasterTask():
    pv_inner_pos = pv_inner.getCoordinates()
    np.savetxt("pos.inner.txt", pv_inner_pos)

# TEST: object_belonging.capsule
# cd object_belonging
# rm -rf pos.inner.txt belonging.out.txt
# mir.run --runargs "-n 1" ./capsule.py
# cat pos.inner.txt | LC_ALL=en_US.utf8 sort > belonging.out.txt
