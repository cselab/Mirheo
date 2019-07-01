#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kbounds', type=float, default=0.0)
parser.add_argument('--ktwist', type=float, default=0.0)
parser.add_argument('--kbending', type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument('--l0_factor', type=float, default=1.0)
parser.add_argument('--tau0', type=float, default=0.0)
parser.add_argument('--tau0_eq', type=float, default=0.0)
parser.add_argument('--center_line', type=str, choices=["helix", "line"])
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt = 1e-3

u = mir.mirheo(ranks, tuple(domain), dt, debug_level=8, log_filename='log', no_splash=True)

com_q = [[ 8., 8., 8.,    1.0, 0.0, 0.0, 0.0]]

L = 5.0
P = 1.0
R = 1.0

if args.center_line == "helix":
    def center_line(s):
        t = s * L * np.pi / P 
        return (R * np.cos(t),
                R * np.sin(t),
                (s-0.5) * L)

    def torsion(s):
        a = R
        b = P / (2.0 * np.pi)
        return b / (a**2 + b**2) + args.tau0_eq

elif args.center_line == "line":
    def center_line(s):
        L = 5.0
        return (0, 0, (s-0.5) * L)

    def torsion(s):
        return args.tau0_eq

def length(a, b):
    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2)
        
num_segments = 100

h = 1.0 / num_segments
l0 = length(center_line(h), center_line(0))

rv = mir.ParticleVectors.RodVector('rod', mass=1, num_segments = num_segments)
ic = mir.InitialConditions.Rod(com_q, center_line, torsion, l0)
u.registerParticleVector(rv, ic)

l0 = l0 * args.l0_factor

prms = {
    "a0" : l0,
    "l0" : l0,
    "k_s_center" : args.kbounds,
    "k_s_frame"  : args.kbounds,
    "k_bending"  : tuple(args.kbending),
    "k_twist"    : args.ktwist,
    "tau0"       : args.tau0,
    "kappa0"     : (0.0, 0.0)
}

int_rod = mir.Interactions.RodForces("rod_forces", **prms);
u.registerInteraction(int_rod)
u.setInteraction(int_rod, rv, rv)

dump_every = 1

u.registerPlugins(mir.Plugins.createForceSaver("forceSaver", rv))
u.registerPlugins(mir.Plugins.createDumpParticles('rod_dump', rv, dump_every, [["forces", "vector"]], 'h5/rod-'))

u.run(2)

del u

# nTEST: rod.forces.bounds
# cd rod
# rm -rf h5
# mir.run --runargs "-n 2" ./forces.py \
# --center_line "helix" --kbounds 1000.0 --l0_factor 1.05
# mir.post ../membrane/utils/post.forces.py --file h5/rod-00001.h5 --out forces.out.txt

# nTEST: rod.forces.twist
# cd rod
# rm -rf h5
# mir.run --runargs "-n 2" ./forces.py \
# --center_line "line" --ktwist 1000.0 --tau0_eq 0.1
# mir.post ../membrane/utils/post.forces.py --file h5/rod-00001.h5 --out forces.out.txt

# nTEST: rod.forces.twist.tau0
# cd rod
# rm -rf h5
# mir.run --runargs "-n 2" ./forces.py \
# --center_line "line" --ktwist 1000.0 --tau0 0.2 --tau0_eq 0.1
# mir.post ../membrane/utils/post.forces.py --file h5/rod-00001.h5 --out forces.out.txt

# nTEST: rod.forces.bending
# cd rod
# rm -rf h5
# mir.run --runargs "-n 2" ./forces.py \
# --center_line "helix" --kbending 1000.0 0.0 1000.0
# mir.post ../membrane/utils/post.forces.py --file h5/rod-00001.h5 --out forces.out.txt
