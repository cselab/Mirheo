#!/usr/bin/env python

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=int)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = (2, 2, 4)

u = mir.Mirheo(ranks, domain, dt=0, debug_level=3, log_filename='log', no_splash = True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(args.density)
u.registerParticleVector(pv, ic)

dump_every = 1

u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv, dump_every, [], 'h5/solvent_particles-'))

u.run(1)

# TEST: dump.h5.parts
# cd dump
# rm -rf h5 h5.parts.out.txt
# mir.run --runargs "-n 2" ./h5.parts.py --density 3
# mir.post h5dump -d position h5/solvent_particles-00000.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8 sort > h5.parts.out.txt

# TEST: dump.h5.parts.noparts
# set -eu
# cd dump
# rm -rf h5 h5.parts.out.txt
# mir.run --runargs "-n 2" ./h5.parts.py --density 0
# mir.post h5dump -d position h5/solvent_particles-00000.h5 | awk '{print $2, $3, $4}' | LC_ALL=en_US.utf8 sort > h5.parts.out.txt
