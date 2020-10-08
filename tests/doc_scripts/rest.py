#!/usr/bin/env python

import mirheo as mir

dt = 0.001
rc = 1.0      # cutoff radius
number_density = 8.0

ranks  = (1, 1, 1)
domain = (16.0, 16.0, 16.0)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1.0) # Create a simple Particle Vector (PV) named 'pv'
ic = mir.InitialConditions.Uniform(number_density)        # Specify uniform random initial conditions
u.registerParticleVector(pv, ic)                          # Register the PV and initialize its particles

# Create and register DPD interaction with specific parameters and cutoff radius
dpd = mir.Interactions.Pairwise('dpd', rc, kind="DPD", a=10.0, gamma=10.0, kBT=1.0, power=0.5)
u.registerInteraction(dpd)

# Tell the simulation that the particles of pv interact with dpd interaction
u.setInteraction(dpd, pv, pv)

# Create and register Velocity-Verlet integrator
vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)

# This integrator will be used to advance pv particles
u.setIntegrator(vv, pv)

# Write some simulation statistics on the screen every 500 time steps
u.registerPlugins(mir.Plugins.createStats('stats', every=500))

# Dump particle data
dump_every = 500
u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv, dump_every, [], 'h5/solvent_particles-'))

u.run(5002, dt=dt)
