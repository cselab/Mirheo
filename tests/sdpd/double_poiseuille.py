#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)
ext_force = 1.0
rc = 1.0

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1.0)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

den  = ymr.Interactions.Density('den', rc, kernel="WendlandC2")
sdpd = ymr.Interactions.SDPD('sdpd', rc, viscosity=10.0, kBT=1.0, EOS="Linear", sound_speed=10.0, density_kernel="WendlandC2")
u.registerInteraction(den)
u.registerInteraction(sdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(sdpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=ext_force, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size,
                                                [("velocity", "vector_from_float8")],
                                                'h5/solvent-'))

u.run(5002)

# nTEST: sdpd.double_poiseuille
# cd sdpd
# rm -rf h5
# ymr.run --runargs "-n 2" ./double_poiseuille.py > /dev/null
# ymr.avgh5 xz velocity h5/solvent-0000[2-4].h5 | awk '{print $1}' > profile.out.txt
