#!/usr/bin/env python

import mirheo as mir
import numpy as np

dt   = 0.001
axes = (1, 2, 3)

ranks  = (1, 1, 1)
domain = (8, 16, 24)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

com_q = [[domain[0]/2.0, domain[1]/2.0, domain[2]/2.0,  1., 0, 0, 0]]
coords = np.loadtxt('sphere123.txt').tolist()
pv_ell = mir.ParticleVectors.RigidEllipsoidVector('object', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = mir.InitialConditions.Rigid(com_q, coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_ell, ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
u.registerParticleVector(pv, mir.InitialConditions.Uniform(number_density=8))

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=2.0, gamma=10.0, kBT=0.1, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)
u.setInteraction(dpd, pv_ell, pv)

vv = mir.Integrators.VelocityVerlet_withConstForce('vv', force=(0.1, 0, 0))
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverageRelative(
    'field', [pv], pv_ell, 0,
    sample_every, dump_every, bin_size, ["velocities"], 'h5/solvent-'))

u.run(5010)


# nTEST: plugins.relative_flow
# set -eu
# cd plugins
# rm -rf h5/
# mir.run --runargs "-n 2" ./relative_flow.py
# mir.avgh5 xy velocities h5/solvent-0000[2-4].h5 >  profile.out.txt
# mir.avgh5 yz velocities h5/solvent-0000[2-4].h5 >> profile.out.txt
# mir.avgh5 zx velocities h5/solvent-0000[2-4].h5 >> profile.out.txt
