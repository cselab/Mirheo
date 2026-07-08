#!/usr/bin/env python

import argparse
import mirheo as mir

parser = argparse.ArgumentParser()
parser.add_argument('--debug-level', type=int)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

rc = 1.0
density = 8

kwargs = {'debug_level': args.debug_level} if args.debug_level is not None else {}
u = mir.Mirheo(ranks, domain, no_splash=True, log_filename='stdout', **kwargs)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic = mir.InitialConditions.Uniform(density)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.run(50, dt=dt)

# Test: (a) env=3, (b) env=0, (c) env=3, --debug-level=0, (d) env=0, --debug-level=3.

# TEST: log.env_var
# cd log
# MIRHEO_DEBUG_LEVEL=3 mir.run --runargs "-n 2" ./env_var.py > env_var.log
# ([ -s env_var.log ] && echo "not empty, ok" || echo "empty, error") > env_var.out.txt
# MIRHEO_DEBUG_LEVEL=0 mir.run --runargs "-n 2" ./env_var.py > env_var.log
# ([ -s env_var.log ] && echo "not empty, error" || echo "empty, ok") >> env_var.out.txt
# MIRHEO_DEBUG_LEVEL=3 mir.run --runargs "-n 2" ./env_var.py --debug-level=0 > env_var.log
# ([ -s env_var.log ] && echo "not empty, error" || echo "empty, ok") >> env_var.out.txt
# MIRHEO_DEBUG_LEVEL=0 mir.run --runargs "-n 2" ./env_var.py --debug-level=3 > env_var.log
# ([ -s env_var.log ] && echo "not empty, ok" || echo "empty, error") >> env_var.out.txt
# echo "end" >> env_var.out.txt
