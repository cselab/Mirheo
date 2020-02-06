#!/usr/bin/env python

"""
Test dumping of Mesh and ObjectVector (particle data + object data).

Currently without any actual particles. Only the XMF structure is tested.
"""

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ranks", type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (4, 6, 8)
dt = 0.1

u = mir.Mirheo(args.ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh1 = mir.ParticleVectors.MembraneMesh('mesh_dummy1.off')
mesh2 = mir.ParticleVectors.MembraneMesh('mesh_dummy2.off')
ov1 = mir.ParticleVectors.MembraneVector('ov1', mesh=mesh1, mass=1)
ov2 = mir.ParticleVectors.MembraneVector('ov2', mesh=mesh2, mass=1)
ic = mir.InitialConditions.Membrane([])
u.registerParticleVector(pv=ov1, ic=ic)
u.registerParticleVector(pv=ov2, ic=ic)

u.writeSnapshot('snapshot/')

# def read_file(filename):
#     with open(filename) as f:
#         return f.read()
#
# # Note: These two assertions will fail if the numbers are not formatted in the same way.
# assert read_file("snapshot/mesh_0.off") == read_file("mesh_dummy1.off")
# assert read_file("snapshot/mesh_1.off") == read_file("mesh_dummy2.off")

# TEST: snapshot.membrane_vectors
# cd snapshot
# rm -rf snapshot/
# mir.run --runargs "-n 4" ./membrane_vectors.py --ranks 2 1 1
# cat snapshot/config.compute.json snapshot/config.post.json snapshot/mesh_?.off snapshot/ov?.PV.xmf snapshot/ov?.OV.xmf > snapshot.out.txt
