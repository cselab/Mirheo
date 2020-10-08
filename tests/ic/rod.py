#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--initial_frame", type=float, required=False, nargs=3)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

com_q = [[ 1., 0., 0.,    1.0, 0.0, 0.0, 0.0],
         [ 5., 0., 0.,    1.0, 2.0, 0.0, 0.0],
         [-9., 0., 0.,    1.0, 0.0, 3.0, 0.0], # out of the domain
         [ 0., 7., 0.,    1.0, 0.0, 0.0, 1.0]]

a = 0.1

def center_line(s):
    L = 5.0
    P = 1.0
    R = 1.0
    t = s * L * np.pi / P
    return (R * np.cos(t),
            R * np.sin(t),
            (s-0.5) * L)

def torsion(s):
    return 0.0

num_segments = 100
rv = mir.ParticleVectors.RodVector('rod', mass=1, num_segments=num_segments)
if args.initial_frame is None:
    ic = mir.InitialConditions.Rod(com_q, center_line, torsion, a)
else:
    ic = mir.InitialConditions.Rod(com_q, center_line, torsion, a, args.initial_frame)
u.registerParticleVector(rv, ic)

dump_every = 1
u.registerPlugins(mir.Plugins.createDumpParticles('rod_dump', rv, dump_every, [], 'h5/rod_particles-'))

u.run(2, dt=0)

if rv:
    icpos = np.array(rv.getCoordinates())
    icvel = np.array(rv.getVelocities())
    num_rods = int(len(icpos) / (5*num_segments))

    nv = num_segments * 5 + 1
    assert nv * num_rods == len(icpos)

    # sort by center of mass x coordinates
    com = np.array([np.mean(icpos[i*nv:(i+1)*nv,:], axis=0) for i in range(num_rods)])

    order = np.argsort(com[:,0])
    idx = np.repeat(order, nv) * nv + np.tile(np.arange(nv), num_rods)
    icpos = icpos[idx,:]
    icvel = icpos[idx,:]

    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)

del u

# nTEST: ic.rod
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./rod.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort  | uscale 100 > ic.out.txt

# nTEST: ic.rod.initial_frame
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./rod.py --initial_frame 1 0 0
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort | uscale 100 > ic.out.txt
