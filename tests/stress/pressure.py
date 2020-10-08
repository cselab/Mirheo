#!/usr/bin/env python

import argparse
import mirheo as mir
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (32, 32, 32)
tdump_every = 0.001
dump_every = int(tdump_every / dt)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

pv_name="pv"
path="pressure"

pv = mir.ParticleVectors.ParticleVector(pv_name, mass = 1)
ic = mir.InitialConditions.Uniform(number_density=10)
u.registerParticleVector(pv, ic)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5, stress=True, stress_period=tdump_every)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

def predicate_all_domain(r):
    return 1.0

h = (1.0, 1.0, 1.0)

u.registerPlugins(mir.Plugins.createVirialPressurePlugin('Pressure', pv, predicate_all_domain, h, dump_every, path))

u.run(2001, dt=dt)

volume = domain[0]*domain[1]*domain[2]

if u.isMasterTask():
    df = pd.read_csv(os.path.join(path, pv_name+".csv"))
    p_mean = np.mean(df["pressure"]) / volume
    np.savetxt(args.out, [p_mean])

del(u)

# nTEST: stress.pressure
# cd stress
# rm -rf pressure pressure.txt
# mir.run --runargs "-n 2" ./pressure.py --out pressure.txt
# cat pressure.txt | uscale 0.1 > pressure.out.txt
