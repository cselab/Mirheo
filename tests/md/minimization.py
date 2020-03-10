#!/usr/bin/env python

from mpi4py import MPI
import mirheo as mir
import numpy as np
import sys

from utils import get_h5_forces

# Units. 1 == Mirheo unit.
nm = 1
fs = 1
kg = 1e27

m = 1e9 * nm
s = 1e15 * fs
kJ = 1e3 * kg * m ** 2 / s ** 2

# Argon and system properties.
epsilon = 0.996 * kJ / 6.022e23
sigma = 0.340 * nm
mass = 39.948 * 1.66053906660e-27 * kg

dt = 0.01 * fs
max_displacement = 0.001 * nm
number_density = 0.6 / sigma ** 3
domain = (12 * nm, 10 * nm, 8 * nm)

u = mir.Mirheo((1, 1, 1), domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass=mass)
ic = mir.InitialConditions.Uniform(number_density=number_density)
u.registerParticleVector(pv, ic)

lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='LJ', epsilon=epsilon, sigma=sigma)
u.registerInteraction(lj)
u.setInteraction(lj, pv, pv)

vv = mir.Integrators.Minimize('minimize', max_displacement=max_displacement)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(mir.Plugins.createForceSaver('forceSaver', pv))
u.registerPlugins(mir.Plugins.createDumpParticles(
        'meshdump', pv, 10, ['forces'], 'h5/pv-'))

u.run(101)

if MPI.COMM_WORLD.rank == 0:
    for i in range(10):
        forces = get_h5_forces('h5/pv-{:05}.h5'.format(i))
        forces = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2 + forces[:, 2]**2)
        forces = np.sum(forces)
        # Using log because nTEST has a tolerance of 0.1.
        print(i, np.log(forces + 1e-9) * 100, file=sys.stderr)

# nTEST: md.minimization
# cd md
# mir.run --runargs "-n 2" ./minimization.py > /dev/null 2> forces.out.txt
