.. _user-pv:

Particle Vectors
################

A :any:`ParticleVector` (or PV) is a collection of particles in the simulation with identical properties.
PV is the minimal unit of particles that can be addressed by most of the processing utilities,
i.e. it is possible to specify interactions between different (or same) PVs,
apply integrators, plugins, etc. to the PVs.

Each particle in the PV keeps its coordinate, velocity and force.
Additional quantities may also be stored in a form of extra channels.
These quantities are usually added and used by specific handlers, and can in principle be written
in XDMF format (:any:`Average3D`), see more details in the Developer documentation.

A common special case of a :any:`ParticleVector` is an :any:`ObjectVector` (or OV).
The OV **is** a Particle Vector with the particles separated into groups (objects) of the same size.
For example, if a single cell membrane is represented by say 500 particles, an object vector
consisting of the membranes will contain all the particles of all the membranes, grouped by 
membrane.
Objects are assumed to be spatially localized, so they always fully reside within a single
MPI process. OV can be used in most of the places where a regular PV can be used, and more


Summary
========
.. automodsumm:: _mirheo.ParticleVectors

.. automod-diagram:: _mirheo.ParticleVectors
    :skip: Mesh, MembraneMesh

Details
========
.. automodule:: _mirheo.ParticleVectors
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__
      
 
