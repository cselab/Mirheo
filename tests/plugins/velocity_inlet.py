#!/usr/bin/env python

import mirheo as mir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["sphere", 'cylinder', 'plane'])
args = parser.parse_args()

dt  = 0.001
kBT = 0.0

ranks  = (1, 1, 1)
domain = (32, 32, 32)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)
u.registerParticleVector(pv, mir.InitialConditions.Uniform(density=0))

vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

h = 0.5
resolution = (h, h, h)

center = (0.5 * domain[0],
          0.5 * domain[1],
          0.5 * domain[2])

radius = 4.0
vel = 10.0
inlet_density = 10

if args.geometry == "sphere":
    def inlet_surface(r):
        R = np.sqrt((r[0] - center[0])**2 +
                    (r[1] - center[1])**2 +
                    (r[2] - center[2])**2)
        return R - radius

    def inlet_velocity(r):
        factor = vel / radius
        return (factor * (r[0] - center[0]),
                factor * (r[1] - center[1]),
                factor * (r[2] - center[2]))

elif args.geometry == "cylinder":
    def inlet_surface(r):
        R = np.sqrt((r[0] - center[0])**2 +
                    (r[1] - center[1])**2)
        return R - radius

    def inlet_velocity(r):
        factor = vel / radius
        x = r[0] - center[0]
        y = r[1] - center[1]
        z = r[2] - center[2]
        factor *= 1 - (z/5.)**2
        factor = max([factor, 0.])
        return (factor * x, factor * y, 0)

elif args.geometry == "plane":
    # plane perpendicular to y direction with velocity in circular patch or radius R
    def inlet_surface(r):
        return r[1] - center[1]
    
    def inlet_velocity(r):
        x = r[0] - center[0]
        z = r[2] - center[2]
        R = 2.0
        U = 1 - (x**2 + z**2) / R**2
        U = vel * max([U, 0.])
        return (0, U, 0)

else:
    exit(1)
    
u.registerPlugins(mir.Plugins.createVelocityInlet('inlet', pv, inlet_surface, inlet_velocity, resolution, inlet_density, kBT))

sample_every = 10
dump_every = 1000
bin_size = (1.0, 1.0, 1.0)
u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [], 'h5/solvent-'))

u.run(1010)

del (u)

# nTEST: plugins.velocity_inlet.sphere
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./velocity_inlet.py --geometry sphere  > /dev/null
# mir.avgh5 yz density h5/solvent-0000*.h5 > profile.out.txt

# nTEST: plugins.velocity_inlet.cylinder
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./velocity_inlet.py --geometry cylinder  > /dev/null
# mir.avgh5 yz density h5/solvent-0000*.h5 > profile.out.txt

# nTEST: plugins.velocity_inlet.plane
# cd plugins
# rm -rf h5
# mir.run --runargs "-n 2" ./velocity_inlet.py --geometry plane  > /dev/null
# mir.avgh5 xz density h5/solvent-0000*.h5 > profile.out.txt
