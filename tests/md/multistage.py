#!/usr/bin/env python

from mpi4py import MPI
import mirheo as mir
import numpy as np
import sys

from utils import get_h5_forces

def main():
    # Units. 1 == Mirheo unit.
    nm = 1
    fs = 1
    kg = 1e27
    K = 1

    m = 1e9 * nm
    s = 1e15 * fs
    J = kg * m ** 2 / s ** 2

    # Argon and system properties.
    epsilon = 996. * J / 6.022e23
    sigma = 0.340 * nm
    mass = 39.948 * 1.66053906660e-27 * kg
    kB = 1.380649e-23 * J / K

    max_displacement = 0.005 * nm
    number_density = 0.1 / sigma ** 3  # Use very small density for testing.
    domain = (12 * nm, 10 * nm, 8 * nm)

    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=mass)
    ic = mir.InitialConditions.Uniform(number_density=number_density)
    u.registerParticleVector(pv, ic)

    lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='LJ', epsilon=epsilon, sigma=sigma)
    u.registerInteraction(lj)
    u.setInteraction(lj, pv, pv)

    ##############################
    # Stage 1.
    ##############################
    int_min = mir.Integrators.Minimize('minimize', max_displacement=max_displacement)
    u.registerIntegrator(int_min)
    u.setIntegrator(int_min, pv)

    # Measure forces, they should decrease over time.
    plugin_force_saver = mir.Plugins.createForceSaver('forceSaver', pv)
    plugin_dump = mir.Plugins.createDumpParticles('meshdump', pv, 200, ['forces'], 'h5/pv-')
    u.registerPlugins(plugin_force_saver)
    u.registerPlugins(plugin_dump)
    u.run(1001, dt=0.01 * fs)

    u.deregisterPlugins(plugin_dump)
    u.deregisterPlugins(plugin_force_saver)
    u.deregisterIntegrator(int_min)
    del plugin_dump
    del plugin_force_saver

    ##############################
    # Stage 2.
    ##############################
    int_vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(int_vv)
    u.setIntegrator(int_vv, pv)

    # Measure temperature, it should approach the reference temperature.
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], kBT=215 * K * kB, tau=50.0 * fs))
    u.registerPlugins(mir.Plugins.createStats('stats', every=50, filename='stats.csv'))
    u.run(500, dt=1 * fs)

    ##############################
    # Stage 3.
    ##############################
    u.run(200, dt=5 * fs)

    if MPI.COMM_WORLD.rank == 0:
        print("log(force) during minimization", file=sys.stderr)
        for i in range(5):
            forces = get_h5_forces('h5/pv-{:05}.h5'.format(i))
            forces = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2 + forces[:, 2]**2)
            forces = np.sum(forces)
            # Using log because nTEST has a tolerance of 0.1.
            print(i, np.log(forces + 1e-9) * 100, file=sys.stderr)

        # This could be done with pandas, but better to avoid importing it.
        print("temperature during equilibration and the run", file=sys.stderr)
        with open('stats.csv', 'r') as f:
            header = f.readline().split(',')
            assert header[0:2] == ['time', 'kBT']
            mat = np.loadtxt(f, delimiter=',', usecols=(0, 1))
            mat[:, 1] /= kB
            for row in mat:
                print(*row, file=sys.stderr)


main()


# nTEST: md.multistage
# cd md
# mir.run --runargs "-n 2" ./multistage.py > /dev/null 2> multistage.out.txt
