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
See :ref:`dev-interactions-membrane-trikernels`, :ref:`dev-interactions-membrane-dihkernels` and :ref:`dev-interactions-membrane-filter` for possible template parameters.

.. doxygenclass:: mirheo::MembraneInteraction
   :project: mirheo
   :members:

.. _dev-interactions-membrane-trikernels:

Triangle Kernels
----------------

Each thread is mapped to one vertex `v1` and loops over all adjacent triangles labeled as follows:

.. graphviz::
   
    graph triangle {
    node [shape=plaintext]
    {rank = same; v2; v1}
    v3 -- v2
    v3 -- v1
    v2 -- v1
    }

The output of the kernel is the forces of a given dihedral on `v1`.
The forces on `v2` and `v3` from the same dihedral are computed by the thread mapped on `v2` and `v3`, respectively.

.. doxygenclass:: mirheo::TriangleLimForce
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::TriangleWLCForce
   :project: mirheo
   :members:


.. _dev-interactions-membrane-dihkernels:

Dihedral Kernels
----------------

Each thread is mapped to one vertex `v0` and loops over all adjacent dihedrals labeled as follows:

.. graphviz::
   
    graph dihedral {
    node [shape=plaintext]
    {rank = same; v2 ; v0}
    v3 -- v2
    v3 -- v0
    v2 -- v0
    v2 -- v1
    v0 -- v1
    }

The output of the kernel is the forces of a given dihedral on `v0` and `v1`.
The forces on `v2` and `v3` from the same dihedral are computed by the thread mapped on `v3`.

.. doxygenclass:: mirheo::DihedralJuelicher
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::DihedralKantor
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

.. _dev-interactions-membrane-fetchers:

Fetchers
--------

Fetchers are used to load generic data that is needed for kernel computation.
In most cases, only vertex coordinates are sufficient (see :any:`mirheo::VertexFetcher`).
Additional data attached to each vertex may be needed, such as mean curvature in e.g. :any:`mirheo::DihedralJuelicher` (see :any:`mirheo::VertexFetcherWithMeanCurvatures`).

.. doxygenclass:: mirheo::VertexFetcher
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::VertexFetcherWithMeanCurvatures
   :project: mirheo
   :members:

