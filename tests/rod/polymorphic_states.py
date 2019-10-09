#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fraction', type=float, default=0.0)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt = 1e-3
t_end = 10
t_dump_every = 1.0
L = 5.0
num_segments = 60

u = mir.Mirheo(ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

com_q = [[ 8., 8., 8.,    1.0, 0.0, 0.0, 0.0]]

a0     = 0.05
tau0   = [0.2, 0.]
kappa0 = [(0.0, 0.0), (0., 0.1)]
E0     = [0.0, 0.0]
drag   = 5.0

def center_line(s):
    return (0, 0, (s-0.5) * L)

def torsion(s):
    if s < args.fraction:
        return tau0[0]
    return  tau0[1]

def length(a, b):
    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2)

h = 1.0 / num_segments
l0 = length(center_line(h), center_line(0))

rv = mir.ParticleVectors.RodVector('rod', mass=1, num_segments = num_segments)
ic = mir.InitialConditions.Rod(com_q, center_line, torsion, a0)
u.registerParticleVector(rv, ic)

prms = {
    "a0" : a0,
    "l0" : l0,
    "k_s_center" : 10000.0,
    "k_s_frame"  : 10000.0,
    "k_bending"  : (30.0, 0.0, 30.0),
    "k_twist"    : 30.0,
    "k_smoothing": 0.0,
    "tau0"       : tau0,
    "kappa0"     : kappa0,
    "E0"         : E0
}

int_rod = mir.Interactions.RodForces("rod_forces", "smoothing", **prms);
u.registerInteraction(int_rod)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setInteraction(int_rod, rv, rv)
u.setIntegrator(vv, rv)

u.registerPlugins(mir.Plugins.createParticleDrag('rod_drag', rv, drag))

dump_every = int (t_dump_every/dt)
u.registerPlugins(mir.Plugins.createDumpParticlesWithRodData('rod_dump', rv, dump_every, [("states", "scalar")], 'h5/rod_particles-'))

u.run(int (t_end / dt))

if rv is not None:
    pos = rv.getCoordinates()
    np.savetxt("pos.rod.txt", pos)

del u

# nTEST: rod.polymorphic_states.0.4
# cd rod
# rm -rf h5
# mir.run --runargs "-n 2" ./polymorphic_states.py --fraction 0.4
# cat pos.rod.txt > pos.out.txt
