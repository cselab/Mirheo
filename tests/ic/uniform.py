#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=float, nargs=3, default=[4., 2., 3.])
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = args.domain
density = 8

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

u.run(2, dt=0)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)
    

# TEST: ic.uniform
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./uniform.py
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt

# TEST: ic.uniform.no_integer_domain
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./uniform.py --domain 4.33 2.5 3.9
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
