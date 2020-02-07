#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, required=True)
parser.add_argument('--write', type=str, required=True)
parser.add_argument('--read', type=str)
args = parser.parse_args()

domain = (4, 6, 8)
dt = 0.1

if not args.read:
    u = mir.Mirheo(args.ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1)
    ic = mir.InitialConditions.Uniform(number_density=2)
    u.registerParticleVector(pv, ic)

    dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)

    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    u.writeSnapshot(args.write)
else:
    u = mir.Mirheo(args.ranks, snapshot=args.read, debug_level=3, log_filename='log', no_splash=True)
    u.writeSnapshot(args.write)

# TEST: snapshot.interactions
# cd snapshot
# echo rm -rf snapshot1/ snapshot2/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --write snapshot1/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1 --write snapshot2/ --read snapshot1/
# git --no-pager diff --no-index snapshot1/config.compute.json snapshot2/config.compute.json
# git --no-pager diff --no-index snapshot1/config.post.json snapshot2/config.post.json
# h5diff snapshot1/pv.PV.h5 snapshot2/pv.PV.h5
# cat snapshot1/config.compute.json snapshot1/config.post.json > snapshot.out.txt
