uDeviceX
########

Mesoscale flow solver for biological and medical applications

Introduction
************

The **uDeviceX** code is designed as a classical molecular dynamics code adapted for inclusion of large (consisting of thousands of particles) rigid bodies and cells.
Main features of the code include:

* fluids represented as free particles interacting with different pairwise potentials (i.e. DPD or Lennard-Jones),
* static walls of arbitrary complexity and size,
* rigid bodies with arbitrary shapes and sizes,
* viscoelastic cell membranes, that can separate inner from outer fluids

The multi-process GPU implementation enables very fast time-to-solution without compromising physical complexity.
Some benchmarks are listed in :ref:`user-bench`.

The following documentation is aimed at providing users a comprehensive simulation guide
as well as exposing code internals for the developers wishing to contribute to the project. 

..  Licensing
..  ---------

.. toctree::
   :maxdepth: 1
   :caption: User guide
   
   user/installation
   user/overview
   user/particle_vectors
   user/initial_conditions
   user/interactions
   user/integrators
   user/walls
   user/bouncers
   user/object_belonging
   user/plugins

   
.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   developer/overview
   developer/basic_types
   developer/coordinators/coordinators
   developer/particle_vectors/particle_vectors
   developer/mpi/mpi
   developer/handlers/handlers
   developer/plugins/plugins
   
..   views
..   parser
..   misc
