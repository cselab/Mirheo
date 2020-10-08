#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)
ext_force = 1.0
rc = 1.0

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1.0)
ic = mir.InitialConditions.Uniform(number_density=10)
u.registerParticleVector(pv=pv, ic=ic)

den  = mir.Interactions.Pairwise('den' , rc, kind="Density", density_kernel="WendlandC2")
sdpd = mir.Interactions.Pairwise('sdpd', rc, kind="SDPD", viscosity=10.0, kBT=1.0, EOS="Linear", sound_speed=10.0, rho_0=0.0, density_kernel="WendlandC2")
u.registerInteraction(den)
u.registerInteraction(sdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(sdpd, pv, pv)

vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=ext_force, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size,
                                                ["velocities"],
                                                'h5/solvent-'))

u.run(5002, dt=dt)

# nTEST: sdpd.double_poiseuille
# cd sdpd
# rm -rf h5
# mir.run --runargs "-n 2" ./double_poiseuille.py
# mir.avgh5 xz velocities h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt
