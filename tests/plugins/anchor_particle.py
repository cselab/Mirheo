#!/usr/bin/env python

import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = [8, 8, 8]

u = mir.Mirheo(ranks, tuple(domain), dt=0.01, debug_level=3, log_filename='log', no_splash=True)

center = (domain[0]/2, domain[1]/2, domain[2]/2)

R = 2.0
Om = 1.0

def positions(t):
    return [(center[0] + R * np.cos(Om * t),
            center[1] + R * np.sin(Om * t),
            0.)]

def velocities(t):
    return [(- R * Om * np.sin(Om * t),
            + R * Om * np.cos(Om * t),
            0.)]

pos = [list(positions(0)[0]), [0., 0., 0.], [1., 1., 1.]]
vel = [[0., 0., 0.]] * 3
pids = [1]

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.FromArray(pos, vel)
u.registerParticleVector(pv, ic)

u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", pv, positions, velocities, pids, 100, "anchor/"))

u.run(500)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)

del u

# TEST: plugins.anchor_particle
# cd plugins
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./anchor_particle.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > pv.out.txt
