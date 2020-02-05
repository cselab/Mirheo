#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ranks", type=int, nargs=3, required=True)
args = parser.parse_args()

domain = (4, 6, 8)
dt = 0.1

u = mir.Mirheo(args.ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass=1)
ic = mir.InitialConditions.Uniform(number_density=2)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)

u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)
u.writeSnapshot('snapshot/')

# TEST: snapshot.interactions
# cd snapshot
# rm -rf snapshot/
# mir.run --runargs "-n 2" ./interactions.py --ranks 1 1 1
# cat snapshot/config.compute.json snapshot/config.post.json > snapshot.out.txt
