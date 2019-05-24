#!/usr/bin/env python

import numpy as np
import ymero as ymr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tau0', type=float, default=0.0)
parser.add_argument('--tau0_init', type=float, default=0.5)
parser.add_argument('--kappa', type=float, nargs=2, default=[0,0])
parser.add_argument('--drag', type=float, default=0.0)
parser.add_argument('--sub_steps', type=int, default=1)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt = 1e-3
t_end = 10
t_dump_every = 1.0
L = 50.0
num_segments = 200
sub_steps = args.sub_steps

u = ymr.ymero(ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

com_q = [[ 8., 8., 8.,    1.0, 0.0, 0.0, 0.0]]

def center_line(s):
    return (0, 0, (s-0.5) * L)

def torsion(s):
    return args.tau0_init

def length(a, b):
    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2)

h = 1.0 / num_segments
l0 = length(center_line(h), center_line(0))

rv = ymr.ParticleVectors.RodVector('rod', mass=1, num_segments = num_segments)
ic = ymr.InitialConditions.Rod(com_q, center_line, torsion, l0)
u.registerParticleVector(rv, ic)

prms = {
    "a0" : l0,
    "l0" : l0,
    "k_s_center": 1000.0,
    "k_s_frame" : 1000.0,
    "k_bending" : (10.0, 0.0, 10.0),
    "k_twist"   : 10.0,
    "tau0"      : args.tau0,
    "kappa0"    : tuple(args.kappa)
}

int_rod = ymr.Interactions.RodForces("rod_forces", **prms);
u.registerInteraction(int_rod)

if sub_steps == 1:
    vv = ymr.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setInteraction(int_rod, rv, rv)
    u.setIntegrator(vv, rv)
else:
    vv = ymr.Integrators.SubStep('vv', sub_steps, int_rod)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, rv)

if args.drag > 0.0:
    u.registerPlugins(ymr.Plugins.createParticleDrag('rod_drag', rv, args.drag))

dump_every = int (t_dump_every/dt)
u.registerPlugins(ymr.Plugins.createDumpParticles('rod_dump', rv, dump_every, [], 'h5/rod_particles-'))

u.run(int (t_end / dt))

if rv is not None:
    pos = rv.getCoordinates()
    np.savetxt("pos.rod.txt", pos)

del u

# nTEST: rod.rest
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py
# cat pos.rod.txt > pos.out.txt

# nTEST: rod.rest.substep
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py --sub_steps 10
# cat pos.rod.txt > pos.out.txt

# nTEST: rod.rest.tau0
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py --tau0 0.5 --tau0_init 0.4
# cat pos.rod.txt > pos.out.txt

# nTEST: rod.rest.helix
# cd rod
# rm -rf h5
# ymr.run --runargs "-n 2" ./rest.py --tau0 0.5 --tau0_init 0.0 --kappa 0.8 0.0 --drag 0.5
# cat pos.rod.txt > pos.out.txt

