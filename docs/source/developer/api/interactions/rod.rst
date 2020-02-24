.. _dev-interactions-rod:

Rod Interactions
================

Base class
----------

This is the visible class that is output of the factory function.

.. doxygenclass:: mirheo::BaseRodInteraction
   :project: mirheo
   :members:

Implementation
--------------

The factory instantiates one of this templated class.

.. doxygenclass:: mirheo::RodInteraction
   :project: mirheo
   :members:

Kernels
-------

The following support structure is used to compute the elastic energy:

.. doxygenstruct:: mirheo::BiSegment
   :project: mirheo
   :members:
