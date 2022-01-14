Mirheo
########

Mesoscale flow solver for biological and medical applications

Introduction
************

Mirheo [alexeev2020]_ is designed as a classical molecular dynamics code adapted for inclusion of large (consisting of thousands of particles) rigid bodies and cells.
The main features of the code include:

* fluids represented as free particles interacting with different pairwise potentials (i.e. DPD or Lennard-Jones),
* static walls of arbitrary complexity and size,
* rigid bodies with arbitrary shapes and sizes [amoudruz2021]_,
* viscoelastic cell membranes [economides2021]_, that can separate inner from outer fluids

The multi-process GPU implementation enables very fast time-to-solution without compromising physical complexity
and Python front-end ensures fast and easy simulation setup.
Some benchmarks are listed in :ref:`user-bench`.

The following documentation is aimed at providing users a comprehensive simulation guide
as well as exposing code internals for the developers wishing to contribute to the project.

.. [alexeev2020] Alexeev, Dmitry, et al. "Mirheo: High-performance mesoscale simulations for microfluidics." Computer Physics Communications 254 (2020): 107298.

.. [economides2021] Economides, Athena, et al. "Hierarchical Bayesian Uncertainty Quantification for a Model of the Red Blood Cell." Physical Review Applied 15.3 (2021): 034062.

.. [amoudruz2021] Amoudruz, Lucas, and Petros Koumoutsakos. "Independent Control and Path Planning of Microswimmers with a Uniform Magnetic Field." Advanced Intelligent Systems (2021): 2100183.


.. toctree::
   :maxdepth: 1
   :caption: User guide

   user/user
   user/installation
   user/testing
   user/tutorials
   user/benchmarks
   user/coordinator
   user/particle_vectors
   user/initial_conditions
   user/object_belonging
   user/integrators
   user/interactions
   user/bouncers
   user/walls
   user/plugins
   user/utils


.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   developer/overview
   developer/coding_conventions
   developer/api
   developer/snapshots
