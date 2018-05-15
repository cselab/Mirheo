.. _user-walls:

Walls
#####

Walls are used to represent time-independent stationary boundary conditions for the flows. 
Walls are represented in the form of a signed distance function (LINK), such that a zero-level isosurface defines the wall surface.
No slip and no through boundary conditions are enforced on that surface by bouncing the particles off the wall surface.

In order to prevent undesired density oscillations near the walls, so called frozen particles are used.
These non-moving particles reside *inside* the walls and interact with the regular liquid particles.
If the density and distribution of the frozen particles is the same as of the corresponding liquid particles,
the density oscillations in the liquid in proximity of the wall is minimal (LINK).

Syntax
******

.. role:: xml(code)
   :language: xml
   
.. role:: bash(code)
   :language: bash

.. code-block:: xml

   <wall attributes="..." >
      <generate_frozen attributes="..." />
      <apply_to pv="name" />
   </wall >

The wall defined by the :xml:`<wall>` node will be applied to all the Particle Vectors that are listed in :xml:`<apply_to>` children of that node.
This means that the particles from the specified PVs will bounce off the wall, preventing penetration.

In the beginning of the simulation all the particles define in the simulation (even not defined in :xml:`<apply_to>`) 
will be checked against all the walls. Those inside the wall as well as objects partly inside the wall will be deleted.
The only exception is the PVs that are named exactly as the wall, these PVs will be unaffected by their "parent" wall.

Generation of the frozen particles
**********************************

Parameters for the generation are listed in the :xml:`<generate_frozen>` node.
Attribute :xml:`density` (float type) governs the number density of the frozen particles,
the other attributes are the same as for the DPD interaction (see :ref:`user-interactions`).

The :bash:`genwall` executable can be used to create the frozen particles.
It should be called with the same input script, and for every wall it will do the following:

#. create the particle vector with the specified number density
#. define a DPD interaction between the particles of the PV according to the parameters of the :xml:`<generate_frozen>` node
#. define a Velocity-Verlet integrator (see :ref:`user-integrators`) for the PV with time-step defined in the parameters of the node
#. run 5000 time-steps with periodic boundrary conditions (no simulation modifications like plugins, etc. take effect here)
#. 


Common attributes
*****************

+-----------+--------+---------+---------------------------------------+
| Attribute | Type   | Default | Remarks                               |
+===========+========+=========+=======================================+
| type      | string | ""      | Type of the plugin, see below for the |
|           |        |         | list of available types               |
+-----------+--------+---------+---------------------------------------+
| name      | string | ""      | Name of the created plugin            |
+-----------+--------+---------+---------------------------------------+

Available Plugins
*****************

* **Add force**
   This plugin will add constant force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
   Is is advised to only use it with rigid objects, since Velocity-Verlet integrator with constant pressure can do the same without any performance penalty.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| pv_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| force     | float3 | (0, 0, 0) | :math:`\mathbf{F}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_force" name="frc"  pv_name="sphere" force="0.1 0 0" />
      