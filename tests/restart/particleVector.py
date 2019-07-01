#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

ranks  = args.ranks
domain = (4, 6, 8)
dt = 0

comm = MPI.COMM_WORLD

u = mir.mirheo(ranks, domain, dt, comm_ptr = MPI._addressof(comm),
              debug_level=3, log_filename='log', no_splash=True,
              checkpoint_every = (0 if args.restart else 5))

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)

if args.restart:
    ic = mir.InitialConditions.Restart("restart/")
else:
    ic = mir.InitialConditions.Uniform(density=2)

u.registerParticleVector(pv, ic)

u.run(7)

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
# mir.run --runargs "-n 1" ./particleVector.py --ranks 1 1 1
# mir.run --runargs "-n 1" ./particleVector.py --ranks 1 1 1 --restart
# cat parts.txt | LC_ALL=en_US.utf8 sort > parts.out.txt

# TEST: restart.particleVector.mpi
# cd restart
# rm -rf restart parts.out.txt parts.txt
# mir.run --runargs "-n 4" ./particleVector.py --ranks 1 2 2
# mir.run --runargs "-n 4" ./particleVector.py --ranks 1 2 2 --restart
# cat parts.txt | LC_ALL=en_US.utf8 sort > parts.out.txt

