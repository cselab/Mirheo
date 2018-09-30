.. _user-all:

Overview
##########

This section describes the uDeviceX interface and introduces the reader to installing and running the code.

The uDeviceX code is designed as a classical molecular dynamics code adapted for inclusion of rigid bodies and cells.
The simulation consists of multiple time-steps during which the particles and bodies will be displaces following laws of mechanics and hydrodynamics.
One time-step roughly consists of the following steps:

* compute all the forces in the system, which are mostly pairwise forces between different particles,
* move the particles by integrating the equations of motions,
* bounce particles off the wall surfaces so that they cannot penetrate the wall even in case of soft-core interactions,
* bounce particles off the bodies (i.e. rigid bodies and elastic membranes),
* perform additional operations dictated by plug-ins (modifications, statistics, data dumps, etc.).

Python interface
*****************

The code uses Python scripting language for the simulation setup.
The script defines simulation domain, number of MPI ranks to run; data containers, namely :ref:`user-pv` and data handlers: 
:ref:`user-ic`, :ref:`user-integrators`, :ref:`user-interactions`, :ref:`user-walls`, :ref:`user-bouncers`, :ref:`user-belongers` and :ref:`user-plugins`.

The setup script usually starts with importing the module, e.g.:

.. code-block:: python

    import udevicex as udx

The coordinator class, :any:`udevicex`, and several submodules will be available after that:

* :ref:`ParticleVectors <user-pv>`.
    Consists of classes that store the collections of particles or objects
    like rigid bodies or cell membranes.
    The handlers from the other submodules usually work with one or several :any:`ParticleVector`.
    Typically classes of this submodule define liquid, cell membranes, rigid objects in the flow, etc.
* :ref:`InitialConditions <user-ic>`.
    Provides various ways of creating initial distributions of particles
    or objects of a :any:`ParticleVector`.
* :ref:`BelongingCheckers <user-belongers>`.
    Provides a way to create a new :any:`ParticleVector` by splitting
    an existing one. The split is based on a given :any:`ObjectVector`: all the particles that were
    inside the objects will form one :any:`ParticleVector`, all the outer particles -- the other 
    :any:`ParticleVector`. Removing inner or outer particles is also possible.
    Typically, that checker will be used to remove particles of fluid from within the suspended bodies,
    or to create a :any:`ParticleVector` describing cytoplasm of cells.
    See also :any:`applyObjectBelongingChecker`.
* :ref:`Interactions <user-interactions>`.
    Various interactions that govern forces between particles.
    Pairwise force-fields (DPD, Lennard-Jones) and membrane forces are available.
* :ref:`Integrators <user-integrators>`.
    Various integrators used to advance particles' coordinates and velocities.
* :ref:`Walls <user-walls>`.
    Provides ways to create various static obstacles in the flow, like a sphere,
    pipe, cylinder, etc. See also :any:`makeFrozenWallParticles`
* :ref:`Bouncers <user-bouncers>`.
    Provides ways to ensure that fluid particles don't penetrate inside of
    objects (or the particles from inside of membranes don't leak out of them).
* :ref:`Plugins <user-plugins>`.
    Some classes from this submodule may influence simulation in one way or
    another, e.g. adding extra forces, adding or removing particles, and so on. Other classes are
    used to write simulation data, like particle trajectories, averaged flow-fields, object coordinates, etc.
    
    
A simple script may look this way:
   
.. role:: bash(code)
   :language: bash

.. literalinclude:: ../../../tests/doc_scripts/basic.py



Running the simulation
**********************

uDeviceX is intended to be executed within MPI environments, e.g.:

.. code-block:: bash

    mpirun -np 12 python3 script.py

The code employs simple domain decomposition strategy (see :any:`udevicex`) with the work
mapping fixed in the beginning of the simulation.

.. warning:: 
    When the simulation is started, every subdomain will have 2 MPI tasks working on it. One of
    the tasks, referred to as *compute task* does the simulation itself, another one (*postprocessing task*)
    is used for asynchronous data-dumps and postprocessing.

.. note::
     Recommended strategy is to place two tasks per single compute node with one GPU or 2 tasks
     per one GPU in multi-GPU configuration. The postprocessing tasks will not use any GPU calls,
     so you may not need multiprocess GPU mode or MPS.

.. note::
    If the code is started with number of tasks exactly equal to the number specified in the script,
    the postprocessing will be disabled. All the plugins that use the postprocessing will not work
    (all the plugins that write anything, for example). This execution mode is mainly aimed at debugging.

The running code will produce several log files (one per MPI task): see :any:`udevicex`.
Most errors in the simulation setup (like setting a negative particle mass) will be reported to the log.
In case the code finishes unexpectedly, the user is advised to take a look at the log.