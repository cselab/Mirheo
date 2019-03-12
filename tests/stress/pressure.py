#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (32, 32, 32)
tdump_every = 0.001
dump_every = int(tdump_every / dt)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv_name="pv"
path="pressure"

pv = ymr.ParticleVectors.ParticleVector(pv_name, mass = 1)
ic = ymr.InitialConditions.Uniform(density=10)
u.registerParticleVector(pv=pv, ic=ic)

dpd = ymr.Interactions.DPDWithStress('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5, stressPeriod=tdump_every)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

def predicate_all_domain(r):
    return 1.0

h = (1.0, 1.0, 1.0)

u.registerPlugins(ymr.Plugins.createVirialPressurePlugin('Pressure', pv, predicate_all_domain, h, dump_every, path))

u.run(2001)

volume = domain[0]*domain[1]*domain[2]

if u.isMasterTask():
    data = np.loadtxt(path+"/"+pv_name+".txt")
    p_mean = np.mean(data[:,1]) / volume
    np.savetxt(args.out, [p_mean])

del(u)

# nTEST: stress.pressure
# cd stress
# rm -rf pressure
# ymr.run --runargs "-n 2" ./pressure.py --out pressure.txt > /dev/null
# cat pressure.txt | uscale 0.1 > pressure.out.txt

