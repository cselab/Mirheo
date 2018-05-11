.. _user-plugins:

Plugins
#######

Plugins are used to add specific data processing or to modify the regular pipeline in certain way.
However, the functionality they provide is not considered essential.

If the simulation is started without postprocess part (see :ref:`overview`), some plugins may be disabled. 

Syntax
******

.. role:: xml(code)
   :language: xml

.. code-block:: xml

   <plugin attributes="..." >
   </plugin >

   
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
   This plugin will add constant force :math:`\mathbf{F}_{extra}` onto each particle of a specific PV every time-step.
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
      
* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` onto each *object* of a specific OV every time-step.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| ov_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| torque    | float3 | (0, 0, 0) | :math:`\mathbf{T}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_torque" name="trq"  pv_name="screw" torque="0 0 100000" />  
      

* **Wall repulsion**
   This plugin will add force on all the particles that are nearby a specified wall. The motivation of this plugin is as follows.
   The particles of regular PVs are prevented from penetrating into the walls by Wall Bouncers.
   However, using Wall Bouncers with Object Vectors may be undesirable (e.g. in case of a very viscous membrane) or impossible (in case of rigid objects).
   In these cases one can use either strong repulsive potential between the object and the wall particle or alternatively this plugin.
   The advantage of the SDF-based repulsion is that small penetrations won't break the simulation.
   
   The force expression looks as follows:
   
   .. math::
   
      \mathbf{F} = \mathbf{\nabla}_{sdf} \cdot \begin{cases}
         0, & sdf < -h\\
         \min(F_{max}, C (sdf + h)), & sdf \geqslant -h\\
      \end{cases}
   
   Additional attributes:
   
+-----------+--------+---------+------------------+
| Attribute | Type   | Default | Remarks          |
+===========+========+=========+==================+
| pv_name   | string | ""      | Name of the PV   |
+-----------+--------+---------+------------------+
| wall_name | string | ""      | Name of the wall |
+-----------+--------+---------+------------------+
| C         | float  | 0       | :math:`C`        |
+-----------+--------+---------+------------------+
| h         | float  | 0.2     | :math:`h`        |
+-----------+--------+---------+------------------+
| maxForce  | float  | 1000    | :math:`F_{max}`  |
+-----------+--------+---------+------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="wall_repulsion" name="wrep" pv_name="rbcs" wall_name="wall" C="1000" h="0.2" max_force="1000" /> 
      

* **Stats**
   This plugin will report aggregate quantities of all the particles in the simulation:
   total number of particles in the simulation, average temperature and momentum, maximum velocity magnutide of a particle
   and also the mean real time per step in milliseconds.
   
   .. note::
      This plugin is inactive if postprocess is disabled.
   
   Additional attributes:
   
+-----------+------+---------+------------------------------------------------------+
| Attribute | Type | Default | Remarks                                              |
+===========+======+=========+======================================================+
| every     | int  | 1000    | Report to standard output every that many time-steps |
+-----------+------+---------+------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="stats" name="stats" every="100" /> 
      

* **Average flow dumper**
   This plugin will project certain quantities of the particles on the grid (by simple binning), perform time-averaging of the grid and dump it in HDF5 format.
   The quantities of interest are represented as *channels* associated with particles vectors.
   Some interactions, integrators, etc. and more notable plug-ins can add to the Particle Vectors per-particles arrays to hold different values.
   These arrays are called *channels*.
   Any such channel may be used in this plug-in, however, user must explicitely specify the type of values that the channel holds.
   Particle number density is used to correctly average the values, so it will be sampled and written in any case.
   
   Additional attributes:
   
+--------------+---------+---------+------------------------------------------------------------------------+
| Attribute    | Type    | Default | Remarks                                                                |
+==============+=========+=========+========================================================================+
| pv_name      | string  | ""      | Name of the PV                                                         |
+--------------+---------+---------+------------------------------------------------------------------------+
| sample_every | integer | 50      | Sample quantities every this many time-steps                           |
+--------------+---------+---------+------------------------------------------------------------------------+
| dump_every   | integer | 5000    | Write files every this many time-steps                                 |
+--------------+---------+---------+------------------------------------------------------------------------+
| bin_size     | float3  | (1,1,1) | Bin size for sampling. The resulting quantities will be cell-centered. |
+--------------+---------+---------+------------------------------------------------------------------------+

   The quantities of interest have to be defined in one or more :xml:`channel` nodes with the following attributes:

   +-----------+--------+---------+----------------------------------------------------------------------------------------------------+
   | Attribute | Type   | Default | Remarks                                                                                            |
   +===========+========+=========+====================================================================================================+
   | name      | string | ""      | Channel name. Always available channels are:                                                       |
   |           |        |         |                                                                                                    |
   |           |        |         | * "velocity" with type "float8"                                                                    |
   |           |        |         | * "force" with type "float4"                                                                       |
   |           |        |         |                                                                                                    |
   +-----------+--------+---------+----------------------------------------------------------------------------------------------------+
   | type      | string | ""      | Provide type of quantity to extract from the channel.                                              |
   |           |        |         | Type can also define a simple transformation from the channel internal structure                   |
   |           |        |         | to the datatype supported in HDF5 (i.e. scalar, vector, tensor)                                    |
   |           |        |         | Available types are:                                                                               |
   |           |        |         |                                                                                                    |
   |           |        |         | * scalar: 1 float per particle                                                                     |
   |           |        |         | * vector: 3 floats per particle                                                                    |
   |           |        |         | * vector_from_float4: 4 floats per particle. 3 first floats will form the resulting vector         |
   |           |        |         | * vector_from_float8 8 floats per particle. 5th, 6th, 7th floats will form the resulting vector.   |
   |           |        |         |   This type is primarity made to be used with velocity since it is stored together with            |
   |           |        |         |   the coordinates as 8 consecutive float numbers: (x,y,z) coordinate, followed by 1 padding value  |
   |           |        |         |   and then (x,y,z) velocity, followed by 1 more padding value                                      |
   |           |        |         | * tensor6: 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz                 |
   |           |        |         |                                                                                                    |
   +-----------+--------+---------+----------------------------------------------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="dump_avg_flow" name="avg"
         pv_name="dpd" path="xdmf/dpd"
         sample_every="10" dump_every="10000" bin_size="1 0.5 2"  >
         
         <channel name="velocity" type="vector_from_float8" />
         <channel name="force" type="vector_from_float4" />
      </plugin>  
      

* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` onto each *object* of a specific OV every time-step.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| ov_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| torque    | float3 | (0, 0, 0) | :math:`\mathbf{T}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_torque" name="trq"  pv_name="screw" torque="0 0 100000" />  
      

* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` onto each *object* of a specific OV every time-step.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| ov_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| torque    | float3 | (0, 0, 0) | :math:`\mathbf{T}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_torque" name="trq"  pv_name="screw" torque="0 0 100000" />  
      

* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` onto each *object* of a specific OV every time-step.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| ov_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| torque    | float3 | (0, 0, 0) | :math:`\mathbf{T}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_torque" name="trq"  pv_name="screw" torque="0 0 100000" />  
      

* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` onto each *object* of a specific OV every time-step.
   
   Additional attributes:
   
+-----------+--------+-----------+----------------------------+
| Attribute | Type   | Default   | Remarks                    |
+===========+========+===========+============================+
| ov_name   | string | ""        | Name of the PV             |
+-----------+--------+-----------+----------------------------+
| torque    | float3 | (0, 0, 0) | :math:`\mathbf{T}_{extra}` |
+-----------+--------+-----------+----------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="add_torque" name="trq"  pv_name="screw" torque="0 0 100000" />  
      

