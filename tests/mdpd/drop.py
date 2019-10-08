#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

R = 6.0
density = 5.0

def ic_filter(r):
    return (r[0] - 0.5*domain[0])**2 + (r[1] - 0.5*domain[1])**2 / 0.5 + (r[2] - 0.5*domain[2])**2 < R**2


pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.UniformFiltered(density, ic_filter)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0
rd = 0.75

den  = mir.Interactions.Pairwise('den', rd, kind="Density", density_kernel="MDPD")
mdpd = mir.Interactions.Pairwise('mdpd', rc, kind="MDPD", rd=rd, a=-40.0, b=40.0, gamma=10.0, kbt=0.5, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

sample_every = 5
dump_every = 5000
bin_size = (0.5, 0.5, 0.5)
u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [], 'h5/solvent-'))

u.run(5001)

del(u)

# nTEST: mdpd.drop
# cd mdpd
# rm -rf stats.txt
# mir.run --runargs "-n 2" ./drop.py
# mir.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt

