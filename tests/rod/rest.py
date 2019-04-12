#!/usr/bin/env python

import numpy as np
import ymero as ymr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tau0', type=float, default=0.0)
parser.add_argument('--tau0_init', type=float, default=0.5)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt = 1e-3

u = ymr.ymero(ranks, tuple(domain), dt, debug_level=8, log_filename='log', no_splash=True)

com_q = [[ 8., 8., 8.,    1.0, 0.0, 0.0, 0.0]]

L = 5.0

def center_line(s):
    L = 5.0
    return (0, 0, (s-0.5) * L)

def torsion(s):
    return args.tau0_init

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
l0 = length(center_line(h), center_line(0))

prms = {
    "a0" : l0,
    "l0" : l0,
    "k_bounds"  : 1000.0,
    "k_bending" : 10.0,
    "k_twist"   : 10.0,
    "tau0"      : args.tau0,
    "omega0_x"  : 0.0,
    "omega0_y"  : 0.0
}

int_rod = ymr.Interactions.RodForces("rod_forces", **prms);
u.registerInteraction(int_rod)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)

u.setInteraction(int_rod, rv, rv)
u.setIntegrator(vv, rv)

dump_every = 10
u.registerPlugins(ymr.Plugins.createDumpParticles('rod_dump', rv, dump_every, [], 'h5/rod_particles-'))

u.run(502)

if rv is not None:
    pos = rv.getCoordinates()
    np.savetxt("pos.rod.txt", pos)

del u

# nTEST: rod.rest
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py > /dev/null
# cat pos.rod.txt > pos.out.txt

# nTEST: rod.rest.tau0
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py --tau0 0.5 --tau0_init 0.4 > /dev/null
# cat pos.rod.txt > pos.out.txt
