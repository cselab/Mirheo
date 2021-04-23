#!/usr/bin/env python

"""
Test dumping of ObjectBelongingCheckers.
"""

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
parser.add_argument('--save-to', type=str, required=True)
parser.add_argument('--load-from', type=str)
args = parser.parse_args()

domain = (4, 6, 8)

if not args.load_from:
    u = mir.Mirheo(args.ranks, domain, debug_level=3, log_filename='log', no_splash=True)

    mesh = mir.ParticleVectors.MembraneMesh('mesh_dummy1.off')
    ov = mir.ParticleVectors.MembraneVector('ov', mesh=mesh, mass=1)
    ic = mir.InitialConditions.Membrane([[1.0, 2.0, 3.0,  1.0, 0.0, 0.0 ,0.0]])
    u.registerParticleVector(ov, ic)

    pv_outer = mir.ParticleVectors.ParticleVector('pv_outer', mass=1.0)
    u.registerParticleVector(pv_outer, mir.InitialConditions.FromArray(pos=[], vel=[]))

    inner_checker = mir.BelongingCheckers.Mesh('inner_solvent_checker')
    u.registerObjectBelongingChecker(inner_checker, ov)
    pv_inner = u.applyObjectBelongingChecker(inner_checker, pv_outer, correct_every=100, inside='pv_inner')

    u.saveSnapshot(args.save_to)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.load_from, debug_level=3, log_filename='log', no_splash=True)
    u.saveSnapshot(args.save_to)
    u.run(1, dt=0.1)  # Test that it does not crash.

# TEST: snapshot.object_belonging
# cd snapshot
# rm -rf snapshot1/ snapshot2/ snapshot.out.txt
# mir.run --runargs "-n 2" ./object_belonging.py --ranks 1 1 1 --save-to snapshot1/
# mir.run --runargs "-n 2" ./object_belonging.py --ranks 1 1 1 --save-to snapshot2/ --load-from snapshot1/
# git --no-pager diff --no-index snapshot1/config.json snapshot2/config.json
# cat snapshot1/config.json > snapshot.out.txt
