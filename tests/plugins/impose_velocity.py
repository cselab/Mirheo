#!/usr/bin/env python

import udevicex as udx
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--velocity', type=float, nargs=3, required=True)
parser.add_argument('--omega',    type=float, nargs=3, required=True)
parser.add_argument('--solvent', action='store_true')

args = parser.parse_args()

dt   = 0.001
axes = (1, 2, 3)

ranks  = (1, 1, 1)
domain = (8, 16, 24)

u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

com_q = [[1, 2, 3,   1., 0, 0, 0],
         [5, 10, 15, 1,  1, 1, 1]]
coords = np.loadtxt('sphere123.txt').tolist()
pvEllipsoid = udx.ParticleVectors.RigidEllipsoidVector('object', mass=1, object_size=len(coords), semi_axes=axes)
icEllipsoid = udx.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoid = udx.Integrators.RigidVelocityVerlet("ellvv", dt)

u.registerParticleVector(pv=pvEllipsoid, ic=icEllipsoid)
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)

if args.solvent:
    pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
    u.registerParticleVector(pv, udx.InitialConditions.Uniform(density=8))
    
    dpd = udx.Interactions.DPD('dpd', 1.0, a=2.0, gamma=10.0, kbt=0.1, dt=dt, power=0.5)
    u.registerInteraction(dpd)
    u.setInteraction(dpd, pv, pv)
    u.setInteraction(dpd, pvEllipsoid, pv)
    
    vv = udx.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=dt, force=1.0, direction='x')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

velocity = tuple( [x if np.abs(x) < 1e3 else udx.Plugins.PinObject.Unrestricted for x in args.velocity] )
omega    = tuple( [x if np.abs(x) < 1e3 else udx.Plugins.PinObject.Unrestricted for x in args.omega] )

u.registerPlugins( udx.Plugins.createPinObject('pin', ov=pvEllipsoid, dump_every=500, path='force/', velocity=velocity, angular_velocity=omega) )
u.registerPlugins( udx.Plugins.createDumpObjectStats("objStats", ov=pvEllipsoid, dump_every=50, path="stats/")
 )

u.run(2010)


# nTEST: plugins.restrictObjects.1
# set -eu
# cd plugins
# rm -rf force/
# udx.run --runargs "-n 2" python impose_velocity.py --velocity 1 0 0 --omega 0 1 0  > /dev/null
# cat stats/*.txt | awk 'NF{NF-=6};1' | sort -n > plugins.out.txt
# cat force/*.txt | sort -n >> plugins.out.txt

# nTEST: plugins.restrictObjects.2
# set -eu
# cd plugins
# rm -rf force/
# udx.run --runargs "-n 2" python impose_velocity.py --velocity 1 0.1 0 --omega 0.1 1 0 --solvent  > /dev/null
# cat stats/*.txt | awk 'NF{NF-=6};1' | sort -n > plugins.out.txt
# cat force/*.txt | sort -n >> plugins.out.txt

# nTEST: plugins.restrictObjects.3
# set -eu
# cd plugins
# rm -rf force/
# udx.run --runargs "-n 2" python impose_velocity.py --velocity 1e10 1e10 0 --omega 1 1e10 1e10 --solvent  > /dev/null
# cat stats/*.txt | awk 'NF{NF-=6};1' | sort -n > plugins.out.txt
# cat force/*.txt | sort -n >> plugins.out.txt
