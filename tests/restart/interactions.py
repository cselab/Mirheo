#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3)
args = parser.parse_args()

ranks  = args.ranks
domain = (4, 6, 8)
dt = 0

if args.restart:
    restart_folder="restart2/"
else:
    restart_folder="restart/"

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', checkpoint_every=5, checkpoint_folder=restart_folder, no_splash=True)
    
pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=2)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

if args.restart:
    u.restart("restart/")
u.run(7)
    

# TEST: restart.interactions
# cd restart
# rm -rf restart restart2 state.out.txt
# mir.run --runargs "-n 1" ./interactions.py --ranks 1 1 1
# mir.run --runargs "-n 1" ./interactions.py --ranks 1 1 1 --restart
# cat restart2/dpd.ParirwiseInt.txt > state.out.txt

