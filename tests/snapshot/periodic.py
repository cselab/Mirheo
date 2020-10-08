#!/usr/bin/env python

"""Test checkpoint-like periodic snapshots.

We test that there are that many folders and that the currentStep changes.
"""

import mirheo as mir

u = mir.Mirheo(nranks=(1, 1, 1), domain=(4, 6, 8), debug_level=3,
               log_filename='log', no_splash=True,
               checkpoint_every=10, checkpoint_mode='Incremental',
               checkpoint_folder='periodic_snapshots/snapshot_', checkpoint_mechanism='Snapshot')

pv = mir.ParticleVectors.ParticleVector('pv', mass=1)
ic = mir.InitialConditions.Uniform(number_density=2)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind='DPD', a=10.0, gamma=10.0, kBT=1.0, power=0.5)
lj = mir.Interactions.Pairwise('lj', rc=1.0, kind='LJ', epsilon=1.25, sigma=0.75)

u.registerInteraction(dpd)
u.registerInteraction(lj)
u.setInteraction(dpd, pv, pv)

minimize = mir.Integrators.Minimize('minimize', max_displacement=1. / 1024)
u.registerIntegrator(minimize)
u.run(45, dt=0.125)

# TEST: snapshot.periodic
# cd snapshot
# rm -rf periodic_snapshots/
# mir.run --runargs "-n 2" ./periodic.py
# ls periodic_snapshots | cat > snapshot.out.txt
# grep -rH --include=*.json currentStep periodic_snapshots/ | sort >> snapshot.out.txt
