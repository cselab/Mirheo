#!/usr/bin/env python

import ymero as ymr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["sphere", 'cylinder'])
args = parser.parse_args()

dt  = 0.001
kBT = 0.0

ranks  = (1, 1, 1)
domain = (32, 32, 32)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
u.registerParticleVector(pv, ymr.InitialConditions.Uniform(density=0))

vv = ymr.Integrators.VelocityVerlet('vv')
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

else:
    exit(1)
    
u.registerPlugins(ymr.Plugins.createVelocityInlet('inlet', pv, inlet_surface, inlet_velocity, resolution, inlet_density, kBT))

#dump_every   = 100
#u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pv, dump_every, [], 'h5/solvent_particles-'))

sample_every = 10
dump_every = 1000
bin_size = (1.0, 1.0, 1.0)
u.registerPlugins(ymr.Plugins.createDumpAverage('field', [pv], sample_every, dump_every, bin_size, [], 'h5/solvent-'))

u.run(1010)

del (u)

# nTEST: plugins.velocityInlet.sphere
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 2" ./velocity_inlet.py --geometry sphere  > /dev/null
# ymr.avgh5 yz density h5/solvent-0000*.h5 > profile.out.txt

# nTEST: plugins.velocityInlet.cylinder
# cd plugins
# rm -rf h5
# ymr.run --runargs "-n 2" ./velocity_inlet.py --geometry cylinder  > /dev/null
# ymr.avgh5 yz density h5/solvent-0000*.h5 > profile.out.txt
