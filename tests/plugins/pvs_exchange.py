#!/usr/bin/env python

import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', type=int, nargs=3, default = (1,1,1))
parser.add_argument('--fraction', type=float, default = 0.5)
args = parser.parse_args()

dt = 0.001
density = 10

ranks  = args.ranks
domain = (16, 16, 16)
a = 1

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

def upHalf(r):
    return r[1] > domain[1] * args.fraction
def loHalf(r):
    return not upHalf(r)

pv1 = mir.ParticleVectors.ParticleVector('pv1', mass = 1)
pv2 = mir.ParticleVectors.ParticleVector('pv2', mass = 1)
ic1 = mir.InitialConditions.UniformFiltered(density, upHalf)
ic2 = mir.InitialConditions.UniformFiltered(density, loHalf)
u.registerParticleVector(pv1, ic1)
u.registerParticleVector(pv2, ic2)
    
dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.1, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv1], sample_every, dump_every, bin_size,
                                                ["velocities"], 'h5/solvent-'))
u.registerPlugins(mir.Plugins.createExchangePVSFluxPlane("color_exchanger", pv1, pv2, (1., 0., 0., 0.)))

u.run(5002)

del(u)

# nTEST: plugins.pvs_exchange
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./pvs_exchange.py
# mir.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvs_exchange.mpi
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 4" ./pvs_exchange.py --ranks 2 1 1
# mir.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvs_exchange.mpi.zero
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 4" ./pvs_exchange.py --ranks 2 1 1 --fraction 0.0
# mir.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

