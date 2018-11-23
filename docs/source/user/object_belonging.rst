.. _user-belongers:

Object belonging checkers
#########################


Object belonging checkers serve two purpooses.
Firstly, they are used to split a :any:`ParticleVector` into two disjointed parts
(probably forming a new Particle Vector): the particles that are *inside* any object of
the given :any:`ObjectVector` and the particles that are *outside*.
See also :any:`ymero.registerObjectBelongingChecker` and :any:`ymero.registerObjectBelongingChecker`.
Secondly, they are used to maintain the mentioned *inside*-*outside* property of the particles
in the resulting :any:`ParticleVectors <ParticleVector>`.
Such maintenance is performed periodically, and the particles of, e.g. inner PV that apper to mistakingly
be outside of the reference :any:`ObjectVector` will be moved to the outer PV (and viceversa).
If one of the PVs was specified as "none", the erroneous particles will be deleted from the simulation.


Summary
========
.. automodsumm:: _ymero.BelongingCheckers

Details
========
.. automodule:: _ymero.BelongingCheckers
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__



 
