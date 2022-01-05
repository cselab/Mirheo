#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default = [1,1,1])
args = parser.parse_args()

dt = 0.001
t_end = 2.0 + dt # time for one restart batch

ranks  = args.ranks
domain = (12, 8, 10)

rc = 1.0
num_density = 8

u = mir.Mirheo(ranks, domain, debug_level=3,
               checkpoint_every = int(1.0/dt), checkpoint_mode="Incremental",
               log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(num_density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', every=500, filename="stats.csv"))

if args.restart:
    u.restart("restart/")

u.run(int(t_end/dt), dt=dt)

# nTEST: restart.plugins.stats
# cd restart
# rm -rf stats.csv restart
# mir.run --runargs "-n 2" ./stats.py > /dev/null
# mir.run --runargs "-n 2" ./stats.py --restart > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt

# nTEST: restart.plugins.stats.mpi
# cd restart
# rm -rf stats.csv restart
# mir.run --runargs "-n 4" ./stats.py --ranks 2 1 1 > /dev/null
# mir.run --runargs "-n 4" ./stats.py --ranks 2 1 1 --restart > /dev/null
# mir.post ../tools/dump_csv.py stats.csv time kBT vx vy vz --header > stats.out.txt
