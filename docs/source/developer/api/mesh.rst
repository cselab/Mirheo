.. _dev-mesh:

Mesh
====

Represent explicit surfaces on the device with triangle mesh.
This was designed for close surfaces.

Internal structure
------------------

A :any:`mirheo::Mesh` is composed of an array of vertices (it contains the coordinates of each vertex) and a list of faces.
Each entry of the faces is composed of three indices that correspond to the vertices in the corresponding triangle.

A :any:`mirheo::MembraneMesh` contains also adjacent information.
This is a mapping from one vertex index to the indices of all adjacent vertices (that share an edge with the input vertex).

Example: In the following mesh, suppose that the maximum degree is ``maxDegree = 7``, the adjacent lists have the entries::

  1 7 8 2 3 * * 0 3 4 5 6 7 * ...

The first part is the ordered list of adjacent vertices of vertex ``0`` (the ``*`` indicates that the entry will not be used).
The second part corresponds to vertex ``1``.
The first entry in each list is arbitrary, only the order is important.
The list of adjacent vertices of vertex ``i`` starts at ``i * maxDegree``.

.. graphviz::

   graph adjacent {
   node [shape=plaintext]
   {rank = same; 3; 4}
   {rank = same; 2; 0; 1; 5}
   {rank = same; 8; 7; 6}
   2 -- 3
   2 -- 0
   2 -- 8
   3 -- 0
   3 -- 4
   3 -- 1
   0 -- 1
   0 -- 8
   0 -- 7
   1 -- 4
   1 -- 5
   1 -- 7
   1 -- 6
   4 -- 5
   8 -- 7
   7 -- 6
   6 -- 5
   }


API
---

Host classes
^^^^^^^^^^^^

.. doxygenclass:: mirheo::Mesh
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::MembraneMesh
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::MeshDistinctEdgeSets
   :project: mirheo
   :members:


Views
^^^^^

.. doxygenstruct:: mirheo::MeshView
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::MembraneMeshView
   :project: mirheo
   :members:
