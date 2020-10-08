#!/usr/bin/env python

"""Test that temperature approaches the target temperature of 215 K.

Test that running with 1 and with 2 ranks produces the same result.
"""

import mirheo as mir
import sys

# Units. 1 == Mirheo unit.
nm = 1
fs = 1
kg = 1e27
K = 1

m = 1e9
s = 1e15
J = kg * m ** 2 / s ** 2


def init():
    # Argon model and system properties.
    epsilon = 996. * J / 6.022e23
    sigma = 0.340 * nm
    mass = 39.948 * 1.66053906660e-27 * kg

    number_density = 0.6 / sigma ** 3
    domain = (10 * nm, 8 * nm, 6 * nm)

    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log',
                   no_splash=True, units=mir.UnitConversion(1 / m, 1 / s, 1 / kg))

    pv = mir.ParticleVectors.ParticleVector('pv', mass=mass)
    ic = mir.InitialConditions.Uniform(number_density=number_density)
    u.registerParticleVector(pv, ic)

    # With the ordinary LJ the simulation would immediately blow up.
    # lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='LJ', epsilon=epsilon, sigma=sigma)
    lj = mir.Interactions.Pairwise('lj', rc=1 * nm, kind='RepulsiveLJ', epsilon=epsilon, sigma=sigma, max_force=1e-4)
    u.registerInteraction(lj)
    u.setInteraction(lj, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=215 * K, tau=50.0 * fs))
    u.registerPlugins(mir.Plugins.createStats('stats', every=50))

    u.run(0, dt=1.0 * fs)
    u.saveSnapshot('snapshot')


def run(nranks):
    u = mir.Mirheo(nranks, snapshot='snapshot', debug_level=3, log_filename='log', no_splash=True)
    u.run(402, dt=1.0 * fs)


if sys.argv[1] == 'init':
    init()
elif sys.argv[1] == 'run':
    run((int(sys.argv[2]), 1, 1))


# nTEST: md.berendsen_thermostat
# cd md
# mir.run --runargs "-n 2" ./berendsen_thermostat.py init
# mir.run --runargs "-n 2" ./berendsen_thermostat.py run 1 | grep Temperature | sed 's/.*(\(.*\) K)/\1/g' > temperature1.txt
# mir.run --runargs "-n 4" ./berendsen_thermostat.py run 2 | grep Temperature | sed 's/.*(\(.*\) K)/\1/g' > temperature2.txt
# git diff --no-index temperature1.txt temperature2.txt
# cat temperature1.txt > temperature.out.txt
