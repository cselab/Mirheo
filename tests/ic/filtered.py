#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filter", choices=["half", "quarter"])
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [4., 4., 4.]
density = 8

u = mir.Mirheo(ranks, tuple(domain), debug_level=3, log_filename='log', no_splash=True)

if args.filter == "half":
    def my_filter(r):
        return r[0] < domain[0] / 2
elif args.filter == "quarter":
    def my_filter(r):
        return r[0] < domain[0] / 2 and r[1] < domain[1] / 2
else:
    exit(1)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.UniformFiltered(density, my_filter)
u.registerParticleVector(pv=pv, ic=ic)

u.run(2, dt=0)

if pv:
    icpos = pv.getCoordinates()
    icvel = pv.getVelocities()
    np.savetxt("pos.ic.txt", icpos)
    np.savetxt("vel.ic.txt", icvel)

del(u)


# TEST: ic.uniform.filtered.half
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./filtered.py --filter half
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt

# TEST: ic.uniform.filtered.quarter
# cd ic
# rm -rf pos*.txt vel*.txt
# mir.run --runargs "-n 2" ./filtered.py --filter quarter
# paste pos.ic.txt vel.ic.txt | LC_ALL=en_US.utf8 sort > ic.out.txt
