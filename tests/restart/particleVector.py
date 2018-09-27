#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse

import  mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

ranks  = args.ranks
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if args.restart and pv:
    color = 1
else:
    color = 0
comm = comm.Split(color, rank)


if args.restart and pv:
    ids = pv.get_indices()
    pos = pv.getCoordinates()
    vel = pv.getVelocities() 

    data = np.hstack((np.atleast_2d(ids).T, pos, vel))
    data = comm.gather(data, root=0)

    if comm.Get_rank() == 0:
        data = np.concatenate(data)
        np.savetxt("parts.txt", data)
    

# TEST: restart.particleVector
# cd restart
# rm -rf restart parts.out.txt parts.txt
# udx.run --runargs "-n 1" ./particleVector.py --ranks 1 1 1           > /dev/null
# udx.run --runargs "-n 1" ./particleVector.py --ranks 1 1 1 --restart > /dev/null
# cat parts.txt | sort > parts.out.txt

# TEST: restart.particleVector.mpi
# cd restart
# rm -rf restart parts.out.txt parts.txt
# udx.run --runargs "-n 4" ./particleVector.py --ranks 1 2 2           > /dev/null
# udx.run --runargs "-n 4" ./particleVector.py --ranks 1 2 2 --restart > /dev/null
# cat parts.txt | sort > parts.out.txt

