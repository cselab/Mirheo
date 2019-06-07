#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (4, 6, 8)

u = ymr.ymero(ranks, domain, dt=0, debug_level=3,
              log_filename='log', no_splash=True,
              checkpoint_every = (0 if args.restart else 5))
    
pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
u.registerParticleVector(pv=pv, ic=ymr.InitialConditions.Uniform(density=8))

coords = [[-1, -1, -1], [1, 1, 1]]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
ov = ymr.ParticleVectors.RigidEllipsoidVector('ov', mass=1, object_size=len(coords), semi_axes=(1,1,1))
u.registerParticleVector(ov, ymr.InitialConditions.Rigid(com_q, coords))

checker = ymr.BelongingCheckers.Ellipsoid('checker')
u.registerObjectBelongingChecker(checker, ov)
inner = u.applyObjectBelongingChecker(checker, pv, inside='inner')

if args.restart:
    u.restart("restart/")
u.run(7)

if u.isComputeTask():
    ids = inner.get_indices()
    pos = inner.getCoordinates()
    vel = inner.getVelocities() 

    data = np.hstack((np.atleast_2d(ids).T, pos, vel))
    data = data[data[:,0].argsort()]
        
    if args.restart:
        initials = np.loadtxt("initial.txt")
        np.savetxt("parts.out.txt", data - initials)
    else:
        np.savetxt("initial.txt", data)
    

# nTEST: restart.object_belonging
# cd restart
# rm -rf restart parts.out.txt initial.txt difference.txt
# ymr.run --runargs "-n 2" ./belonging_checker.py
# ymr.run --runargs "-n 2" ./belonging_checker.py --restart
