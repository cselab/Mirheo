.. _user-walls:

Walls
#####

Walls are used to represent time-independent stationary boundary conditions for the flows.
They are described in the form of a `signed distance function <https://en.wikipedia.org/wiki/Signed_distance_function>`_,
such that a zero-level isosurface defines the wall surface.
No slip and no through boundary conditions are enforced on that surface by bouncing the particles off the wall surface.

In order to prevent undesired density oscillations near the walls, so called frozen particles are used.
These non-moving particles reside inside the walls and interact with the regular liquid particles.
If the density and distribution of the frozen particles is the same as of the corresponding liquid particles,
the density oscillations in the liquid in proximity of the wall is minimal.
The frozen particles have to be created based on the wall in the beginning of the simulation,
see :any:`Mirheo.makeFrozenWallParticles`.

In the beginning of the simulation all the particles defined in the simulation
(even not attached to the wall by :any:`Mirheo`) will be checked against all of the walls.
Those inside the wall as well as objects partly inside the wall will be deleted.
The only exception are the frozen PVs, created by the :any:`Mirheo.makeFrozenWallParticles` or
the PVs manually set to be treated as frozen by :any:`Wall.attachFrozenParticles`

Summary
========
.. automodsumm:: mmirheo.Walls

Details
========
.. automodule:: mmirheo.Walls
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__
