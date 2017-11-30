Bouncers
========

Handle collisions of particles and objects.

In general, particles should not penetrate objects or go through membrane separating different liquids.
Bouncers provide a mechanism to maintain that property.
They are called by the end of the simulation pipeline, after the particle integration is done
and the objects are exchanged with the neighbouring MPI ranks.

.. doxygenfile:: bouncers/interface.h

.. toctree::
   ellipsoid_bounce
   mesh_bounce