Overview
========

Task Dependency Graph
---------------------

The simulation is composed of a set of tasks that have dependencies between them,
e.g. the forces must be computed before integrating the particles velocities and positions.
The following graph represents the tasks that are executed at every time step and there dependencies:

.. To reproduce this: use cytoscape with yFiles hierarchical layout
   and export to graphviz format; I had to manually change the node box size
   
.. graphviz:: ../resources/task_graph.dot
   :caption: The task dependency graph of a single time step in Mirheo.
