#!/usr/bin/env python

import ymero as ymr

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)
tdumpEvery = 0.1
dumpEvery = int(tdumpEvery / dt)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPDWithStress('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5, stressPeriod=tdumpEvery)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

def predicate_all_domain(r):
    return 1.0

h = (1.0, 1.0, 1.0)

u.registerPlugins(ymr.Plugins.createVirialPressurePlugin('Pressure', pv, predicate_all_domain, h, dumpEvery, "pressure"))

u.run(1001)

# nTEST: stress.pressure
# cd stress
# rm -rf pressure
# ymr.run --runargs "-n 2" ./pressure.py > /dev/null
# cat pressure/pv.txt | awk '{print $1, 0.01*$2}' > stats.out.txt

