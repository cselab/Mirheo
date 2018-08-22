.. _user-walls:

Walls
#####

Walls are used to represent time-independent stationary boundary conditions for the flows. They are described in the form of a signed distance function (LINK), such that a zero-level isosurface defines the wall surface. No slip and no through boundary conditions are enforced on that surface by bouncing the particles off the wall surface.

In order to prevent undesired density oscillations near the walls, so called frozen particles are used. These non-moving particles reside inside the walls and interact with the regular liquid particles. If the density and distribution of the frozen particles is the same as of the corresponding liquid particles, the density oscillations in the liquid in proximity of the wall is minimal (LINK).

In the beginning of the simulation all the particles define in the simulation (even not attached to the wall by class::udevicex) will be checked against all the walls. Those inside the wall as well as objects partly inside the wall will be deleted. The only exception is the PVs that are named exactly as the wall, these PVs will be unaffected by their “parent” wall.

Summary
========
.. automodsumm:: _udevicex.Walls

Details
========
.. automodule:: _udevicex.Walls
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__









    
