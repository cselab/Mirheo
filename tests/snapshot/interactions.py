#!/usr/bin/env python

import mirheo as mir
import numpy as np
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

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1)
    ic = mir.InitialConditions.Uniform(number_density=2)
    u.registerParticleVector(pv, ic)

    dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)

    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    u.saveSnapshot(args.save_to)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.load_from, debug_level=3, log_filename='log', no_splash=True)
    u.saveSnapshot(args.save_to)

# NOTE: The development docs include this test case as a JSON sample.
#       Currently it assumes it contains only the two JSON objects.

# TEST: snapshot.interactions
# cd snapshot
# rm -rf snapshot1/ snapshot2/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --save-to snapshot1/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --save-to snapshot2/ --load-from snapshot1/
# git --no-pager diff --no-index snapshot1/config.compute.json snapshot2/config.compute.json
# git --no-pager diff --no-index snapshot1/config.post.json snapshot2/config.post.json
# mir.post h5diff snapshot1/pv.PV.h5 snapshot2/pv.PV.h5
# cat snapshot1/config.compute.json snapshot1/config.post.json > snapshot.out.txt
