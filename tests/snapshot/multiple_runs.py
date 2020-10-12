#!/usr/bin/env python

"""
Test that running a simulation with 3x100 time steps produces exactly the same
result as running with 1x300 time steps.

We use a relatively large relative tolerance of 1e-5 to avoid problems with
randomness. For reference, when running with a wrong number of time steps, the
relative error is large (>100%).
"""

import mirheo as mir
import argparse


# Units. 1 == Mirheo unit.
nm = 1
fs = 1
kg = 1e27
K = 1

m = 1e9
s = 1e15
J = kg * m ** 2 / s ** 2
kB = 1.380649e-23 * J /K

# Argon model and system properties.
epsilon = 996. * J / 6.022e23
sigma = 0.340 * nm
mass = 39.948 * 1.66053906660e-27 * kg

parser = argparse.ArgumentParser()
parser.add_argument('--nruns', type=int, required=True)
parser.add_argument('--nsteps', type=int, required=True)
parser.add_argument('--save-to', type=str, required=True)
args = parser.parse_args()

domain = (10, 12, 14)

u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass=mass)
ic = mir.InitialConditions.Uniform(number_density=5)
u.registerParticleVector(pv, ic)

# With the ordinary LJ the simulation would immediately blow up.
# lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='LJ', epsilon=epsilon, sigma=sigma)
lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='RepulsiveLJ', epsilon=epsilon, sigma=sigma, max_force=1e-4)
u.registerInteraction(lj)
u.setInteraction(lj, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createStats('stats', every=50))

for _ in range(args.nruns):
    u.run(args.nsteps, dt=1.0 * fs)

u.saveSnapshot(args.save_to)

# TEST: snapshot.multiple_runs
# cd snapshot
# rm -rf snapshot1/ snapshot2/ snapshot.out.txt
# mir.run --runargs "-n 2" ./multiple_runs.py --nruns=1 --nsteps=300 --save-to=snapshot1/ > /dev/null
# mir.run --runargs "-n 2" ./multiple_runs.py --nruns=3 --nsteps=100 --save-to=snapshot2/ > /dev/null
# git --no-pager diff --no-index snapshot1/config.json snapshot2/config.json
# git --no-pager diff --no-index snapshot1/pv.PV.xmf snapshot2/pv.PV.xmf
# mir.post ../common/hdf5_compare.py compare_pvs --rtol=1e-5 --files snapshot1/pv.PV.h5 snapshot2/pv.PV.h5
# echo "dummy" > snapshot.out.txt
