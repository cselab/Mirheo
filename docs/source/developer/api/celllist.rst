.. _dev-celllist:

Cell-Lists
==========

Cell-lists are used to map from space to particles and vice-versa.

Internal structure
------------------

A cell list is composed of:

#. The representation of the cells geometry (here a uniform grid): see :any:`mirheo::CellListInfo`
#. Number of particles per cell
#. Index of the first particle in each cell
#. The particle data, reordered to match the above structure. 

API
---

.. doxygenclass:: mirheo::CellListInfo
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::CellList
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PrimaryCellList
   :project: mirheo
   :members:

