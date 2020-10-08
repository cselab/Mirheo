#!/usr/bin/env python

"""This file tests interactions and integrators."""

# Note: Almost all numbers below are chosen to be exactly representable, to
# minimize the differences between single and double precision snapshot JSONs.

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
parser.add_argument('--save-to', type=str, required=True)
parser.add_argument('--load-from', type=str)
args = parser.parse_args()

if not args.load_from:
    u = mir.Mirheo(args.ranks, domain=(4, 6, 8),
                   debug_level=15, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1)
    ic = mir.InitialConditions.Uniform(number_density=2)
    u.registerParticleVector(pv, ic)

    dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)
    lj = mir.Interactions.Pairwise('lj', rc=1.0, kind='LJ', epsilon=1.25, sigma=0.75)

    u.registerInteraction(dpd)
    u.registerInteraction(lj)
    u.setInteraction(dpd, pv, pv)

    minimize = mir.Integrators.Minimize('minimize', max_displacement=1. / 1024)
    u.registerIntegrator(minimize)

    u.saveSnapshot(args.save_to)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.load_from, debug_level=3, log_filename='log', no_splash=True)
    u.saveSnapshot(args.save_to)
    u.run(0, dt=0.1)  # Test that it does not crash.

# NOTE: The development docs include this test case as a JSON sample.
#       If updating this test case, check if the docs has to be updated.

# TEST: snapshot.interactions
# cd snapshot
# rm -rf snapshot1/ snapshot2/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --save-to snapshot1/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --save-to snapshot2/ --load-from snapshot1/
# git --no-pager diff --no-index snapshot1/config.json snapshot2/config.json
# mir.post h5diff snapshot1/pv.PV.h5 snapshot2/pv.PV.h5
# cp snapshot1/config.json snapshot.out.txt
