.. _user-belongers:

Object belonging checkers
#########################


Object belonging checkers serve two purpooses:

    1. Split a :any:`ParticleVector` into two disjointed parts (possibly forming a new Particle Vector):
       the particles that are *inside* any object of the given :any:`ObjectVector` and the particles that are *outside*.
    2. Maintain the mentioned *inside*-*outside* property of the particles in the resulting :any:`ParticleVectors <ParticleVector>`.
       Such maintenance is performed periodically, and the particles of, e.g. inner PV that appear to mistakingly be outside of the
       reference :any:`ObjectVector` will be moved to the outer PV (and viceversa).
       If one of the PVs was specified as "none", the erroneous particles will be deleted from the simulation.

See also :any:`Mirheo.registerObjectBelongingChecker` and :any:`Mirheo.applyObjectBelongingChecker`.

Summary
========
.. automodsumm:: mmirheo.BelongingCheckers

Details
========
.. automodule:: mmirheo.BelongingCheckers
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __init__
