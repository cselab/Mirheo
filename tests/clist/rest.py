#!/usr/bin/env python

import argparse
import ymero as ymr

parser = argparse.ArgumentParser()
parser.add_argument("--non_primary", action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (6, 6, 6)

rcFake = 1.0
rc = 0.75

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=4)
u.registerParticleVector(pv=pv, ic=ic)

if args.non_primary:
    null_lj = ymr.Interactions.LJ('fake', rc, epsilon=0.0, sigma=rcFake, object_aware=False)
    u.registerInteraction(null_lj)
    u.setInteraction(null_lj, pv, pv)

dpd = ymr.Interactions.DPD('dpd', rc, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

dump_every = 100

u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", dump_every))

#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, dump_every, [], 'h5/solvent_particles-'))

u.run(501)

# nTEST: clist.primary.rc
# cd clist
# rm -rf stats.txt
# ymr.run --runargs "-n 2" ./rest.py > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

# nTEST: clist.nonPrimary.rc
# cd clist
# rm -rf stats.txt
# ymr.run --runargs "-n 2" ./rest.py --non_primary > /dev/null
# cat stats.txt | awk '{print $1, $2, $3, $4, $5}' > stats.out.txt

