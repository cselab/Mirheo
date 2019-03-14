#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (16, 16, 16)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

R = 6.0
density = 5.0

def ic_filter(r):
    return (r[0] - 0.5*domain[0])**2 + (r[1] - 0.5*domain[1])**2 / 0.5 + (r[2] - 0.5*domain[2])**2 < R**2


pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.UniformFiltered(density, ic_filter)
u.registerParticleVector(pv=pv, ic=ic)

rc = 1.0
rd = 0.75

den  = ymr.Interactions.Density('den', rd, kernel="MDPD")
mdpd = ymr.Interactions.MDPD('mdpd', rc, rd, a=-40.0, b=40.0, gamma=10.0, kbt=0.5, power=0.5)
u.registerInteraction(den)
u.registerInteraction(mdpd)
u.setInteraction(den, pv, pv)
u.setInteraction(mdpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

#u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", 1000))
#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, 1000, [], 'h5/solvent_particles-'))

sampleEvery = 5
dumpEvery = 5000
binSize = (0.5, 0.5, 0.5)
u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [], 'h5/solvent-'))

u.run(5001)

del(u)

# nTEST: mdpd.drop
# cd mdpd
# rm -rf stats.txt
# ymr.run --runargs "-n 2" ./drop.py > /dev/null
# ymr.avgh5 xz density h5/solvent-00000.h5 > profile.out.txt

