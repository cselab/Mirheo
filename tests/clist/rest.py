#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument("--non_primary", action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (6, 6, 6)

rc_fake = 1.0
rc = 0.75

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=4)
u.registerParticleVector(pv, ic)

if args.non_primary:
    null_lj = mir.Interactions.Pairwise('fake', rc, kind="RepulsiveLJ", epsilon=0.0, sigma=rc / (2**(1/6)), max_force=1000.0)
    u.registerInteraction(null_lj)
    u.setInteraction(null_lj, pv, pv)

dpd = mir.Interactions.Pairwise('dpd', rc, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

dump_every = 100

u.registerPlugins(mir.Plugins.createStats('stats', "stats", dump_every))

u.run(501)

# nTEST: clist.primary.rc
# cd clist
# rm -rf stats.csv
# mir.run --runargs "-n 2" ./rest.py > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt

# nTEST: clist.nonPrimary.rc
# cd clist
# rm -rf stats.csv
# mir.run --runargs "-n 2" ./rest.py --non_primary > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt
