#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default = [1,1,1])
args = parser.parse_args()

dt = 0.001

ranks  = args.ranks
domain = (12, 8, 10)
vtarget = (1.0, 0, 0)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True,
               checkpoint_every = 1000)

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

u.registerPlugins(mir.Plugins.createVelocityControl("vc", "vcont.csv", [pv],
                                                    (0, 0, 0), domain,
                                                    5, 5, 50, vtarget,
                                                    Kp, Ki, Kd))


if args.restart:
    u.restart("restart/")

u.run(2501, dt=dt)

# nTEST: restart.vcont
# cd restart
# rm -rf vcont.csv
# mir.run --runargs "-n 2" ./vcont.py
# mir.run --runargs "-n 2" ./vcont.py --restart
# mir.post ../tools/dump_csv.py vcont.csv time vx --header > vcont.out.txt
