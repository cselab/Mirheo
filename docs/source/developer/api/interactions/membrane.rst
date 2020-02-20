.. _dev-interactions-membrane:

Membrane Interactions
=====================

Base class
----------

This is the visible class that is output of the factory function.

.. doxygenclass:: mirheo::BaseMembraneInteraction
   :project: mirheo
   :members:

Implementation
--------------

The factory instantiates one of this templated class.

.. doxygenclass:: mirheo::MembraneInteraction
   :project: mirheo
   :members:

.. _dev-interactions-membrane-filter:

Filters
-------

The membrane interactions can be applied to only a subset of the given :any:`mirheo::MembraneVector`.
This can be convenient to have different interaction parameters for different membranes with the same mesh topology.
Furthermore, reducing the numper of :any:`mirheo::ParticleVector` is beneficial for performance (less interaction kernel launches so overhead for e.g. FSI).

.. doxygenclass:: mirheo::FilterKeepAll
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::FilterKeepByTypeId
   :project: mirheo
   :members:
