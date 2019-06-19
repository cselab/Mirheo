#!/usr/bin/env python

import ymero as ymr
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

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

def upHalf(r):
    return r[1] > domain[1] * args.fraction
def loHalf(r):
    return not upHalf(r)

pv1 = ymr.ParticleVectors.ParticleVector('pv1', mass = 1)
pv2 = ymr.ParticleVectors.ParticleVector('pv2', mass = 1)
ic1 = ymr.InitialConditions.UniformFiltered(density, upHalf)
ic2 = ymr.InitialConditions.UniformFiltered(density, loHalf)
u.registerParticleVector(pv1, ic1)
u.registerParticleVector(pv2, ic2)
    
dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.1, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv1, pv1)
u.setInteraction(dpd, pv1, pv2)
u.setInteraction(dpd, pv2, pv2)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=a, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv1)
u.setIntegrator(vv, pv2)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv1], sample_every, dump_every, bin_size,
                                                [("velocity", "vector_from_float4")], 'h5/solvent-'))
u.registerPlugins(ymr.Plugins.createExchangePVSFluxPlane("color_exchanger", pv1, pv2, (1., 0., 0., 0.)))

u.run(5002)

del(u)

# nTEST: plugins.pvs_exchange
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 2" ./pvs_exchange.py
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvs_exchange.mpi
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 4" ./pvs_exchange.py --ranks 2 1 1
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvs_exchange.mpi.zero
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 4" ./pvs_exchange.py --ranks 2 1 1 --fraction 0.0
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

