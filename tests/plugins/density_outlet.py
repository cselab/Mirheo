#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_density", type=float, required=True)
args = parser.parse_args()

dt  = 0.001

ranks  = (1, 1, 1)
domain = (32, 32, 32)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
u.registerParticleVector(pv, mir.InitialConditions.Uniform(number_density=8))

vv = mir.Integrators.Translate('translate', (10.0, 0.0, 0.0))
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

h = 1.0
resolution = (h, h, h)

def outlet_region(r):
    return -r[0] + (domain[0] - 1.0)

u.registerPlugins(mir.Plugins.createDensityOutlet('outlet', [pv], args.max_density, outlet_region, resolution))

dump_every   = 100
sample_every = 10
bin_size = (1.0, 1.0, 1.0)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [], 'h5/solvent-'))

u.run(1010)

del (u)

# nTEST: plugins.density_outlet.killAll
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./density_outlet.py --max_density 0.0
# mir.avgh5 yz number_densities h5/solvent-00009.h5 > profile.out.txt

