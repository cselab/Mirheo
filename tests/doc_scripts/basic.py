#!/usr/bin/env python
import mirheo as mir

# Simulation time-step
dt = 0.001

# 1 simulation task
ranks  = (1, 1, 1)

# Domain setup
domain = (8, 16, 30)

# Applied extra force for periodic poiseuille flow
f = 1

# Create the coordinator, this should precede any other setup from mirheo package
u = mir.mirheo(ranks, domain, dt, debug_level=2, log_filename='log')

pv = mir.ParticleVectors.ParticleVector('pv', mass = 1)   # Create a simple particle vector
ic = mir.InitialConditions.Uniform(density=8)             # Specify uniform random initial conditions
u.registerParticleVector(pv=pv, ic=ic)                    # Register the PV and initialize its particles

# Create and register DPD interaction with specific parameters
dpd = mir.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, power=0.5)
u.registerInteraction(dpd)

# Tell the simulation that the particles of pv interact with dpd interaction
u.setInteraction(dpd, pv, pv)

# Create and register Velocity-Verlet integrator with extra force
vv = mir.Integrators.VelocityVerlet_withPeriodicForce('vv', force=f, direction='x')
u.registerIntegrator(vv)

# This integrator will be used to advance pv particles
u.setIntegrator(vv, pv)

# Set the dumping parameters
sample_every = 2
dump_every   = 1000
bin_size     = (1., 1., 1.)

# Write some simulation statistics on the screen
u.registerPlugins(mir.Plugins.createStats('stats', every=500))

# Create and register XDMF plugin
u.registerPlugins(mir.Plugins.createDumpAverage('field', [pv],
                                                sample_every, dump_every, bin_size,
                                                [("velocity", "vector_from_float4")], 'h5/solvent-'))

# Run 5002 time-steps
u.run(5002)
