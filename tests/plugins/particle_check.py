#!/usr/bin/env python

import numpy as np
import mirheo as mir

ranks  = (1, 1, 1)
domain = [4.0, 5.0, 6.0]

dt = 100.0 # large dt to force the particles to go too far

u = mir.Mirheo(ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)

pos = [[a*domain[0], a*domain[1], a*domain[2]] for a in [0.1, 0.5, 0.8]]
v=[1., 2., 3.]
vel = [[a*v[0], a*v[1], a*v[2]] for a in [0.0, 0.0, 1.0]] # only one particle will go out of bounds

ic = mir.InitialConditions.FromArray(pos, vel)
u.registerParticleVector(pv, ic)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

check_every = 1
u.registerPlugins(mir.Plugins.createParticleChecker('checker', check_every))

u.run(2)
    

# TEST: plugins.particle_check.bounds
# cd plugins
# rm -rf check*.txt
# mir.run --runargs "-n 2" ./particle_check.py 2>&1 | grep Bad > check.out.txt
