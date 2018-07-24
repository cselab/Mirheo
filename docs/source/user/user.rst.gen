.. _user-all:

Overview
##########

This section describes the uDeviceX interface and introduces the reader to installing and running the code.

The uDeviceX code is designed as a classical molecular dynamics code adapted for inclusion rigid bodies and cells.
The simulation consists of multiple time-steps during which the particles and bodies will be displaces following laws of mechanics and hydrodynamics.
One time-step roughly consists of the following steps:

* compute all the forces in the system, which are mostly pairwise forces between different particles,
* move the particles by integrating the equations of motions,
* bounce particles off the wall surfaces so that they cannot penetrate the wall even in case of soft-core interactions,
* bounce particles off the bodies (i.e. rigid bodies and elastic membranes),
* perform additional operations dictated by plug-ins (modifications, statistics, data dumps, etc.).

Python scripts
***************

The code uses Python scripting language for the simulation setup.
The script defines simulation domain, number of MPI ranks to run; data containers, namely :ref:`user-pv` and data handlers: 
:ref:`user-ic`, :ref:`user-integrators`, :ref:`user-interactions`, :ref:`user-walls`, :ref:`user-bouncers`, :ref:`user-belongers` and :ref:`user-plugins`.
A simple script looks this way:
   
.. role:: bash(code)
   :language: bash

.. code-block:: python

    #!/usr/bin/env python

    import udevicex as udx
    import numpy as np

    # Simulation time-step
    dt = 0.001

    # 1 simulation task
    ranks  = (1, 1, 1)

    # Domain setup
    domain = (8, 16, 16)

    # Applied extra force for periodic poiseuille flow
    f = 1

    # Create the coordinator, this should precede any other setup from udevicex package
    u = udx.udevicex(ranks, domain, debug_level=2, log_filename='log')

    pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)   # Create a simple particle vector
    ic = udx.InitialConditions.Uniform(density=4)             # Specify uniform random initial conditions
    u.registerParticleVector(pv=pv, ic=ic)                    # Register the PV and initialize its particles

    # Create and register DPD interaction with specific parameters
    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
    u.registerInteraction(dpd)

    # Tell the simulation that the particles of pv interact with dpd interaction
    u.setInteraction(dpd, pv, pv)

    # Create and register Velocity-Verlet integrator with extra force
    vv = udx.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=dt, force=f, direction='x')
    u.registerIntegrator(vv)

    # This integrator will be used to advance pv particles
    u.setIntegrator(vv, pv)

    # Set the dumping parameters
    sampleEvery = 2
    dumpEvery   = 1000
    binSize     = (1., 1., 1.)

    # Create and register XDMF plugin
    field = udx.Plugins.createDumpAverage('field', pv, sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
    u.registerPlugins(field)

    # Run 5002 time-steps
    u.run(5002)

Running the simulation
**********************

uDeviceX is intended to be executed within MPI environments, e.g.:

.. code-block:: bash

    mpirun -np 12 python3 script.py
    
You have to submit twice as more MPI tasks as specified in the script (by the nranks parameter of the coordinator udevicex), because every second rank is only responsible for running some plugins and dumping data. Those MPI task are referred to as postprocessing tasks. Recommended strategy is to place two tasks per single compute node with one GPU or 2 tasks pers one GPU in multi-GPU configuration. The postprocessing tasks will not use any GPU calls, so you may not need multiprocess GPU mode or MPS.

If the code is started with number of tasks exactly equal to the number specified in the script, the postprocessing will be disabled. All the plugins that use the postprocessing will not work. This execution mode is mainly aimed at debugging.

 
