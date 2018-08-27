#!/usr/bin/env python

import udevicex as udx

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)
vtarget = (1.0, 0, 0)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=2)
u.registerParticleVector(pv=pv, ic=ic)

dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

vc = udx.Plugins.createVelocityControl("vc", "vcont.txt", [pv], (0, 0, 0), domain, 5, 50, vtarget, Kp, Ki, Kd)
u.registerPlugins(vc)

stats = udx.Plugins.createStats('stats', "stats.txt", 1000)
u.registerPlugins(stats)

u.run(5001)

# nTEST: flow.uniform_vel
# cd flow
# rm -rf vcont.txt
# udx.run --runargs "-n 2" ./uniform_vel.py > /dev/null
# cat vcont.txt | awk '{print $1, $3}' > vcont.out.txt

