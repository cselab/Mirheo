#!/usr/bin/env python

import udevicex as ymr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=int)
args = parser.parse_args()


ranks  = (1, 1, 1)
domain = (2, 2, 4)

u = ymr.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(args.density)
u.registerParticleVector(pv=pv, ic=ic)

dumpEvery   = 1

pvDump = ymr.Plugins.createDumpParticles('partDump', pv, dumpEvery, [], 'h5/solvent_particles-')
u.registerPlugins(pvDump)

u.run(2)

# TEST: dump.h5.parts
# cd dump
# rm -rf h5 h5.parts.out.txt
# ymr.run --runargs "-n 2" ./h5.parts.py --density 3 > /dev/null
# ymr.post h5dump -d position h5/solvent_particles-00000.h5 | awk '{print $2, $3, $4}' | sort > h5.parts.out.txt

# TEST: dump.h5.parts.noparts
# set -eu
# cd dump
# rm -rf h5 h5.parts.out.txt
# ymr.run --runargs "-n 2" ./h5.parts.py --density 0 > /dev/null
# ymr.post h5dump -d position h5/solvent_particles-00000.h5 | awk '{print $2, $3, $4}' | sort > h5.parts.out.txt
