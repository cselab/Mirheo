#!/usr/bin/env python

"""
Test dumping of Mesh and ObjectVector (particle data + object data).

Currently without any actual particles. Only the XMF structure is tested.
"""

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
parser.add_argument('--save-to', type=str, required=True)
parser.add_argument('--load-from', type=str)
args = parser.parse_args()

domain = (4, 6, 8)
dt = 0.1

if not args.load_from:
    u = mir.Mirheo(args.ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

    mesh1 = mir.ParticleVectors.MembraneMesh('mesh_dummy1.off')
    mesh2 = mir.ParticleVectors.MembraneMesh('mesh_dummy2.off')
    ov1 = mir.ParticleVectors.MembraneVector('ov1', mesh=mesh1, mass=1)
    ov2 = mir.ParticleVectors.MembraneVector('ov2', mesh=mesh2, mass=1)
    ic = mir.InitialConditions.Membrane([[1.0, 2.0, 3.0,  1.0, 0.0, 0.0 ,0.0]])
    u.registerParticleVector(ov1, ic)
    u.registerParticleVector(ov2, ic)
    u.saveSnapshot(args.save_to)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.load_from, debug_level=3, log_filename='log', no_splash=True)
    u.saveSnapshot(args.save_to)
    u.run(1)  # Test that it does not crash.

# TODO: Add h5diff once particles (membranes) are added. h5diff does not work
# for empty data sets for some reason.

# TEST: snapshot.membrane_vectors
# cd snapshot
# rm -rf snapshot1/ snapshot2/ snapshot.out.txt
# mir.run --runargs "-n 2" ./membrane_vectors.py --ranks 1 1 1 --save-to snapshot1/
# mir.run --runargs "-n 2" ./membrane_vectors.py --ranks 1 1 1 --save-to snapshot2/ --load-from snapshot1/
# git --no-pager diff --no-index snapshot1/config.compute.json snapshot2/config.compute.json
# git --no-pager diff --no-index snapshot1/config.post.json snapshot2/config.post.json
# git --no-pager diff --no-index snapshot1/mesh_0.off snapshot2/mesh_0.off
# git --no-pager diff --no-index snapshot1/mesh_1.off snapshot2/mesh_1.off
# git --no-pager diff --no-index snapshot1/mesh_0.stressFree.dat snapshot2/mesh_0.stressFree.dat
# git --no-pager diff --no-index snapshot1/mesh_1.stressFree.dat snapshot2/mesh_1.stressFree.dat
# cat snapshot1/config.compute.json snapshot1/config.post.json snapshot1/mesh_?.off snapshot1/ov?.PV.xmf snapshot1/ov?.OV.xmf > snapshot.out.txt
