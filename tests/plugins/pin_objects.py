#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--velocity', type=float, nargs=3, required=True)
parser.add_argument('--omega',    type=float, nargs=3, required=True)
parser.add_argument('--solvent', action='store_true',  default=False)

args = parser.parse_args()

dt   = 0.001
axes = (1, 2, 3)

ranks  = (1, 1, 1)
domain = (8, 16, 24)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

com_q = [[1, 2, 3,   1., 0, 0, 0],
         [5, 10, 15, 1,  1, 1, 1]]

coords = np.loadtxt('sphere123.txt').tolist()
pv_ell = ymr.ParticleVectors.RigidEllipsoidVector('object', mass=1, object_size=len(coords), semi_axes=axes)
ic_ell = ymr.InitialConditions.Rigid(com_q, coords)
vv_ell = ymr.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv_ell, ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

if args.solvent:
    pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
    u.registerParticleVector(pv, ymr.InitialConditions.Uniform(density=8))
    
    dpd = ymr.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.001, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    u.setInteraction(dpd, pv_ell, pv)
    
    vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', force=1.0, direction='x')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

velocity = tuple( [x if np.abs(x) < 1e3 else ymr.Plugins.PinObject.Unrestricted for x in args.velocity] )
omega    = tuple( [x if np.abs(x) < 1e3 else ymr.Plugins.PinObject.Unrestricted for x in args.omega] )

dump_every=50

u.registerPlugins( ymr.Plugins.createPinObject('pin', pv_ell, dump_every, 'force/', velocity, omega) )
u.registerPlugins( ymr.Plugins.createDumpObjectStats("objStats", pv_ell, dump_every, "stats/") )

u.run(2010)


# nTEST: plugins.pin_objects.1
# set -eu
# cd plugins
# rm -rf force/
# ymr.run --runargs "-n 2" python pin_objects.py --velocity 1 0 0 --omega 0 1 0
# cat stats/*.txt | awk 'NF{NF-=6};1' | LC_ALL=en_US.utf8 sort -n > plugins.out.txt
# cat force/*.txt | LC_ALL=en_US.utf8 sort -n >> plugins.out.txt

# nTEST: plugins.pin_objects.2
# set -eu
# cd plugins
# rm -rf force/
# ymr.run --runargs "-n 2" python pin_objects.py --velocity 1 0.1 0 --omega 0.1 1 0 --solvent
# cat stats/*.txt | awk 'NF{NF-=6};1' | LC_ALL=en_US.utf8 sort -n                                                 > plugins.out.txt
# cat force/*.txt | LC_ALL=en_US.utf8 sort -n | awk '{ print $3/500, $4/500, $5/500, $6/5000, $7/5000, $8/5000}' >> plugins.out.txt

# nTEST: plugins.pin_objects.3
# set -eu
# cd plugins
# rm -rf force/
# ymr.run --runargs "-n 2" python pin_objects.py --velocity 1e10 1e10 0 --omega 1 1e10 1e10 --solvent
# cat stats/*.txt | awk 'NF{NF-=6};1' | LC_ALL=en_US.utf8 sort -n               > plugins.out.txt
# cat force/*.txt | LC_ALL=en_US.utf8 sort -n | awk '{ print $5/500, $6/5000}' >> plugins.out.txt
