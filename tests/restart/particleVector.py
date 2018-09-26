#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (4, 6, 8)

if args.restart:
    u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log', checkpoint_every=0)
else:
    u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log', checkpoint_every=5)

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)

if args.restart:
    ic = udx.InitialConditions.Restart("restart/")
else:
    ic = udx.InitialConditions.Uniform(density=2)

u.registerParticleVector(pv=pv, ic=ic)

u.run(7)

if args.restart and pv:
    ids = pv.get_indices()   
    pos = pv.getCoordinates()
    vel = pv.getVelocities() 
    
    np.savetxt("parts.txt", np.hstack((np.atleast_2d(ids).T, pos, vel)))
    

# TEST: restart.particleVector
# cd restart
# rm -rf restart parts.out.txt parts.txt
# udx.run --runargs "-n 1" ./particleVector.py           > /dev/null
# udx.run --runargs "-n 1" ./particleVector.py --restart > /dev/null
# cat parts.txt | sort > parts.out.txt

