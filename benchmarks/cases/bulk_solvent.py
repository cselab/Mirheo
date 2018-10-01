#!/usr/bin/env python

import udevicex as udx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranks', nargs=3, type=int)
parser.add_argument('--domain', nargs=3, type=float)
parser.add_argument('--density', type=float)
parser.add_argument('--out', type=str, default="stats.txt")
args = parser.parse_args()

dt = 0.001

u = udx.udevicex(args.ranks, args.domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(args.density)
u.registerParticleVector(pv, ic)

dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = udx.Integrators.VelocityVerlet('vv', dt)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

stats = udx.Plugins.createStats("stats", args.out, 100)
u.registerPlugins(stats)

u.run(5003)
