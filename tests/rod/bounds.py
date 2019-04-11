#!/usr/bin/env python

import numpy as np
import ymero as ymr

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt = 1e-3

u = ymr.ymero(ranks, tuple(domain), dt, debug_level=8, log_filename='log', no_splash=True)

com_q = [[ 8., 8., 8.,    1.0, 0.0, 0.0, 0.0]]


L = 5.0
P = 1.0
R = 1.0

def center_line(s):
    t = s * L * np.pi / P 
    return (R * np.cos(t),
            R * np.sin(t),
            (s-0.5) * L)

# def center_line(s):
#     L = 5.0
#     return (0, 0, (s-0.5) * L)

# def torsion(s):
#     a = R
#     b = P / (2.0 * np.pi)
#     return b / (a**2 + b**2)

def torsion(s):
    return 0.0

def length(a, b):
    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2)
        
num_segments = 100

rv = ymr.ParticleVectors.RodVector('rod', mass=1, num_segments = num_segments)
ic = ymr.InitialConditions.Rod(com_q, center_line, torsion)
u.registerParticleVector(rv, ic)

h = 1.0 / num_segments
l0 = length(center_line(h), center_line(0)) * 1.05 # small perturbation

prms = {
    "a0" : l0,
    "l0" : l0,
    "k_bounds"  : 1000.0,
    "k_bending" : 0.0,
    "k_twist"   : 0.0
}

int_rod = ymr.Interactions.RodForces("rod_forces", **prms);
u.registerInteraction(int_rod)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)

u.setInteraction(int_rod, rv, rv)
u.setIntegrator(vv, rv)

dump_every = 100
u.registerPlugins(ymr.Plugins.createDumpParticles('rod_dump', rv, dump_every, [], 'h5/rod_particles-'))

# drag = 5.0
# u.registerPlugins(ymr.Plugins.createParticleDrag('drag', rv, drag))

u.run(502)

if rv is not None:
    pos = rv.getCoordinates()
    np.savetxt("pos.rod.txt", pos)

del u

# nTEST: rod.bounds
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./bounds.py > /dev/null
# cat pos.rod.txt > pos.out.txt
