#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)
vtarget = (1.0, 0, 0)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(number_density=2)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

u.registerPlugins(mir.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 5, 50, vtarget, Kp, Ki, Kd))
u.registerPlugins(mir.Plugins.createStats('stats', "stats.csv", 1000))

u.run(5001)

# nTEST: flow.uniform_vel
# cd flow
# rm -rf vcont.txt
# mir.run --runargs "-n 2" ./uniform_vel.py > /dev/null
# cat vcont.txt | awk '{print $1, $3}' > vcont.out.txt
