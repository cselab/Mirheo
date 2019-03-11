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

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

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

sampleEvery = 2
dump_every  = 1000
binSize     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv1], sampleEvery, dump_every, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-'))
u.registerPlugins(ymr.Plugins.createExchangePVSFluxPlane("color_exchanger", pv1, pv2, (1., 0., 0., 0.)))
# u.registerPlugins(ymr.Plugins.createDumpParticles('partDump1', pv1, dump_every, [], 'h5/pv1-'))
# u.registerPlugins(ymr.Plugins.createDumpParticles('partDump2', pv2, dump_every, [], 'h5/pv2-'))

u.run(5002)

del(u)

# nTEST: plugins.pvsExchange
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 2" ./pvsExchange.py > /dev/null
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvsExchange.mpi
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 4" ./pvsExchange.py --ranks 2 1 1 > /dev/null
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

# nTEST: plugins.pvsExchange.mpi.zero
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 4" ./pvsExchange.py --ranks 2 1 1 --fraction 0.0 > /dev/null
# ymr.avgh5 xz density h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt

