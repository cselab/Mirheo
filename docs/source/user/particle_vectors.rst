.. _user-pv:

Particle Vectors
################

:any:`ParticleVector` (or PV) is a collection of particles in the simulation with identical properties. PV is the minimal unit of particles that can be addressed by most of the processing utilities, i.e. it is possible to specify interactions between different (or same) PVs, apply integrators, plugins, etc. to the PVs.

Each particle in the PV keeps its coordinate, velocity and force. Additional quantities may also be stored in a form of extra channels. These quantities are usually added and used by specific handlers, and can in principle be written in XDMF format (Average3D), see more detail in the Developer documentation.

A common special case of the ParticleVector is an :any:`ObjectVector` (or OV), which is a Particle Vector with the particles separated into groups (objects) of the same size. Objects are assumed to be spatially localized, so they always fully reside of a single MPI process. OV can be used in most of the places where a regular PV is used, and more


Summary
========
.. automodsumm:: libudevicex.ParticleVectors

.. automod-diagram:: libudevicex.ParticleVectors

Details
========
.. automodule:: libudevicex.ParticleVectors
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__
      
 
