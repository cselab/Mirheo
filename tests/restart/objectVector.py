#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse
import trimesh

from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

comm   = MPI.COMM_WORLD
ranks  = args.ranks
domain = (16, 16, 16)

if args.restart:
    u = udx.udevicex(MPI._addressof(comm), ranks, domain, debug_level=8, log_filename='log', checkpoint_every=0)
else:
    u = udx.udevicex(MPI._addressof(comm), ranks, domain, debug_level=8, log_filename='log', checkpoint_every=5)

    
mesh = trimesh.creation.icosphere(subdivisions=1, radius = 0.1)
    
udx_mesh = udx.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())
pv       = udx.ParticleVectors.MembraneVector("pv", mass=1.0, mesh=udx_mesh)

if args.restart:
    ic   = udx.InitialConditions.Restart("restart/")
else:
    nobjs = 10
    pos = [ np.array(domain) * t for t in np.linspace(0, 1.0, nobjs) ]
    Q = [ np.array([1.0, 0., 0., 0.])  for i in range(nobjs) ]
    pos_q = np.concatenate((pos, Q), axis=1)

    ic = udx.InitialConditions.Membrane(pos_q.tolist())

u.registerParticleVector(pv, ic)

u.run(7)

rank = comm.Get_rank()

if args.restart and pv:
    color = 1
else:
    color = 0
comm = comm.Split(color, rank)

if args.restart and pv:    
    Pos = pv.getCoordinates()
    Vel = pv.getVelocities()

    data = np.concatenate((Pos, Vel), axis=1)
    data = comm.gather(data, root=0)

    if comm.Get_rank() == 0:
        data = np.concatenate(data)
        np.savetxt("parts.txt", data)
    

# TEST: restart.objectVector
# cd restart
# rm -rf restart parts.out.txt parts.txt
# udx.run --runargs "-n 1" ./objectVector.py --ranks 1 1 1           > /dev/null
# udx.run --runargs "-n 1" ./objectVector.py --ranks 1 1 1 --restart > /dev/null
# cat parts.txt | sort > parts.out.txt

# TEST: restart.objectVector.mpi
# cd restart
# rm -rf restart parts.out.txt parts.txt
# udx.run --runargs "-n 2" ./objectVector.py --ranks 1 1 2           > /dev/null
# udx.run --runargs "-n 2" ./objectVector.py --ranks 1 1 2 --restart > /dev/null
# cat parts.txt | sort > parts.out.txt

