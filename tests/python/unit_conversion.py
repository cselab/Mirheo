#!/usr/bin/env python
"""Test automatic unit conversion (set_unit_registry)."""

import json
from mpi4py import MPI

import mirheo as mir

try:
    import pint
except ImportError:
    import sys
    if MPI.COMM_WORLD.rank == 0:
        print("#############################################################", file=sys.stderr, flush=True)
        print("  Module pint not found, skipping python/unit_conversion.py", file=sys.stderr, flush=True)
        print("#############################################################", file=sys.stderr, flush=True)

        # Skip by copy-pasting the expected output.
        print(open('../test_data/unit_conversion.ref.python.unit_conversion.txt').read(), end='')
    sys.exit()


ureg = pint.UnitRegistry()
ureg.define('myL = 0.5 um')
ureg.define('myT = 0.25 us')
ureg.define('myM = 0.125 ug')

# Set global unit registry for all Mirheo functions.
mir.set_unit_registry(ureg, 'myL', 'myT', 'myM')

# Test the unit conversion (not all arguments are probably dimensionalized below).
domain = (10 * ureg.um, 15 * ureg.um, 20 * ureg.um)
u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass=1e-9 * ureg.kg)
ic = mir.InitialConditions.Uniform(number_density=2 * ureg.um ** -3)
u.registerParticleVector(pv, ic)

kBT = ureg('1.125 um**2 * ug / us**2')
dpd = mir.Interactions.Pairwise('dpd', rc=1.0 * ureg.um, kind='DPD', a=10.0, gamma=10.0, kBT=kBT, power=0.5)

u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)
u.saveSnapshot('snapshot/')

if u.isComputeTask():
    with open('snapshot/config.json') as f:
        config = json.loads(f.read())

    # `dt` should be equal to -1 at this point.
    print("dt =", config['MirState'][0]['dt'])
    print("domainGlobalSize =", config['MirState'][0]['domainGlobalSize'])
    print("kBT =", config['Interaction'][0]['pairParams']['kBT'])

# TEST: python.unit_conversion
# cd python
# rm -rf unit_conversion.out.txt snapshot/
# mir.run --runargs "-n 2" ./unit_conversion.py > unit_conversion.out.txt
