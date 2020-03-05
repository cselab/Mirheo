.. _dev-xdmf:

XDMF
====

A set of classes and functions to write/read data to/from xdmf + hdf5 files format.

Grids
-----

:any:`mirheo::XDMF::Grid` objects are used to represent the geometry of the data that will be dumped.

Interface
^^^^^^^^^

.. doxygenclass:: mirheo::XDMF::GridDims
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::XDMF::Grid
   :project: mirheo
   :members:

Implementation
^^^^^^^^^^^^^^

.. doxygenclass:: mirheo::XDMF::UniformGrid
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::XDMF::VertexGrid
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::XDMF::TriangleMeshGrid
   :project: mirheo
   :members:

