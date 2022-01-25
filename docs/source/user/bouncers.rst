.. _user-bouncers:

Object bouncers
###############

Bouncers prevent particles from crossing boundaries of objects (maintaining no-through boundary conditions).
The idea of the bouncers is to move the particles that crossed the object boundary after integration step back to the correct side.
Particles are moved such that they appear very close (about :math:`10^{-5}` units away from the boundary).
Assuming that the objects never come too close to each other or the walls, this approach ensures that recovered particles will not penetrate into a different object or wall.
In practice maintaining separation of at least :math:`10^{-3}` units between walls and objects is sufficient.
Note that particle velocities are also altered, which means that objects experience extra force from the collisions.

See also :any:`Mirheo.registerBouncer` and :any:`Mirheo.setBouncer`.

Summary
========
.. automodsumm:: mmirheo.Bouncers

Details
========
.. automodule:: mmirheo.Bouncers
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__
