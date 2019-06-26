#!/usr/bin/env python

import mirheo as mir
import numpy as np

density = 4

ranks  = (1, 1, 1)

length = 5.0
radius = 1.0
domain = (8.0, 8.0, 2*length)

u = mir.mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash=True)

def center_line(s): return (0., 0., (s-0.5) * length)
def torsion(s): return 0.0

com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]

pv_rod = mir.ParticleVectors.RodVector('rod', mass=1, num_segments = 10)
ic_rod = mir.InitialConditions.Rod(com_q, center_line, torsion, radius)
u.registerParticleVector(pv_rod, ic_rod)

pv_outer = mir.ParticleVectors.ParticleVector('pv_outer', mass = 1.0)
ic_outer = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv_outer, ic_outer)

inner_checker = mir.BelongingCheckers.Rod("inner_solvent_checker", radius)
u.registerObjectBelongingChecker(inner_checker, pv_rod)

pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every = 0, inside = "pv_inner")

u.run(1)

if u.isMasterTask():
    pv_inner_pos = pv_inner.getCoordinates()
    np.savetxt("pos.inner.txt", pv_inner_pos)

# TEST: object_belonging.rod
# cd object_belonging
# rm -rf pos.inner.txt belonging.out.txt
# mir.run --runargs "-n 1" ./rod.py
# cat pos.inner.txt | LC_ALL=en_US.utf8 sort > belonging.out.txt
