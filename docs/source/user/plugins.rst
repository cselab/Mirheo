.. _user-plugins:

Plugins
#######

Plugins are used to add specific data processing or to modify the regular pipeline in certain way.
However, the functionality they provide is not considered essential.

If the simulation is started without postprocess part (see :ref:`user-overview`), most of the plugins are disabled. 

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
      
* **Add torque**
   This plugin will add constant torque :math:`\mathbf{T}_{extra}` to each *object* of a specific OV every time-step.
   
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
      This plugin is inactive if postprocess is disabled
   
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
   This plugin will project certain quantities of the particles on the grid (by simple binning),
   perform time-averaging of the grid and dump it in XDMF (LINK) format with HDF5 (LINK) backend.
   The quantities of interest are represented as *channels* associated with particles vectors.
   Some interactions, integrators, etc. and more notable plug-ins can add to the Particle Vectors per-particles arrays to hold different values.
   These arrays are called *channels*.
   Any such channel may be used in this plug-in, however, user must explicitely specify the type of values that the channel holds.
   Particle number density is used to correctly average the values, so it will be sampled and written in any case.
   
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+--------------+---------+-------------+--------------------------------------------------------------------------------+
| Attribute    | Type    | Default     | Remarks                                                                        |
+==============+=========+=============+================================================================================+
| pv_name      | string  | ""          | Name of the PV                                                                 |
+--------------+---------+-------------+--------------------------------------------------------------------------------+
| sample_every | integer | 50          | Sample quantities every this many time-steps                                   |
+--------------+---------+-------------+--------------------------------------------------------------------------------+
| dump_every   | integer | 5000        | Write files every this many time-steps                                         |
+--------------+---------+-------------+--------------------------------------------------------------------------------+
| bin_size     | float3  | (1,1,1)     | Bin size for sampling. The resulting quantities will be cell-centered.         |
+--------------+---------+-------------+--------------------------------------------------------------------------------+
| path         | string  | "xdmf/flow" | Path and filename prefix for the dumps                                         |
|              |         |             | For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5 |
+--------------+---------+-------------+--------------------------------------------------------------------------------+

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
      

* **Average relative flow**
   This plugin acts just like the regular flow dumper, with one difference.
   It will assume a coordinate system attached to the center of mass of a specific object.
   In other words, velocities and coordinates sampled correspond to the object reference frame.
   
   .. note::
      Note that this plugin needs to allocate memory for the grid in the full domain, not only in the corresponding MPI subdomain.
      Therefore large domains will lead to running out of memory
      
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| Attribute      | Type    | Default     | Remarks                                                                        |
+================+=========+=============+================================================================================+
| pv_name        | string  | ""          | Name of the PV                                                                 |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| sample_every   | integer | 50          | Sample quantities every this many time-steps                                   |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| dump_every     | integer | 5000        | Write files every this many time-steps                                         |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| bin_size       | float3  | (1,1,1)     | Bin size for sampling. The resulting quantities will be cell-centered.         |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| path           | string  | "xdmf/flow" | Path and filename prefix for the dumps                                         |
|                |         |             | For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5 |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| relative_to_ov | string  | ""          | Take an object governing the frame of reference from this Object Vector        |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
| relative_to_id | integer | 0           | Take an object governing the frame of reference with the specific ID           |
+----------------+---------+-------------+--------------------------------------------------------------------------------+
   
   The quantities of interest have to be defined in one or more :xml:`channel` nodes, exactly like in case of a regular flow dumper.

   **Example**
   
   .. code-block:: xml
   
      <plugin type="dump_avg_relative_flow" name="rel"
         pv_name="dpd" path="xdmf/relative"
         relative_to_ov="sphere" relative_to_id="0"
         sample_every="20" dump_every="10000" bin_size="1 1 1" >
         
         <channel name="velocity" type="vector_from_float8" />
         <channel name="stress"   type="tensor6" />
      </plugin>
      

* **XYZ dumper**
   This plugin will dump positions of all the particles of the specified Particle Vector in the XYZ format.
   
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+------------+---------+---------+---------------------------------------------------------------+
| Attribute  | Type    | Default | Remarks                                                       |
+============+=========+=========+===============================================================+
| pv_name    | string  | ""      | Name of the PV                                                |
+------------+---------+---------+---------------------------------------------------------------+
| dump_every | integer | 1000    | Write every this many time-steps                              |
+------------+---------+---------+---------------------------------------------------------------+
| path       | string  | "xyz/"  | The filenames will look like this: <path>/<pv_name>_NNNNN.xyz |
+------------+---------+---------+---------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="dump_xyz" name="xyz" pv_name="sphere" dump_every="1000" path="xyz/" /> 
      

* **Mesh dumper**
   This plugin will write the meshes of all the object of the specified Object Vector in a PLY format (LINK).
   
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+------------+---------+---------+---------------------------------------------------------------+
| Attribute  | Type    | Default | Remarks                                                       |
+============+=========+=========+===============================================================+
| ov_name    | string  | ""      | Name of the OV                                                |
+------------+---------+---------+---------------------------------------------------------------+
| dump_every | integer | 1000    | Write every this many time-steps                              |
+------------+---------+---------+---------------------------------------------------------------+
| path       | string  | "ply/"  | The filenames will look like this: <path>/<ov_name>_NNNNN.ply |
+------------+---------+---------+---------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="dump_mesh" name="ply" ov_name="cells" dump_every="500" path="ply/" /> 
      

* **Object properties dumper**
   This plugin will write the coordinates of the centers of mass of the objects of the specified Object Vector.
   If the objects are rigid bodies, also will be written: COM velocity, rotation, angular velocity, force, torque.
   
   The file format is the following:
   
   <object id> <simulation time> <COM>x3 [<quaternion>x4 <velocity>x3 <angular velocity>x3 <force>x3 <torque>x3]
   
   .. note::
      Note that all the written values are *instantaneous*
      
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+------------+---------+---------+---------------------------------------------------------------+
| Attribute  | Type    | Default | Remarks                                                       |
+============+=========+=========+===============================================================+
| ov_name    | string  | ""      | Name of the OV                                                |
+------------+---------+---------+---------------------------------------------------------------+
| dump_every | integer | 1000    | Write every this many time-steps                              |
+------------+---------+---------+---------------------------------------------------------------+
| path       | string  | "pos/"  | The filenames will look like this: <path>/<ov_name>_NNNNN.txt |
+------------+---------+---------+---------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="dump_obj_pos" name="position" ov_name="sphere" dump_every="100" path="pos/" />

      
* **Object pinning**
   This plugin will fix center of mass positions (by axis) of all the objects of the specified Object Vector.
   If the objects are rigid bodies, rotatation may be restricted with this plugin as well.
   The *average* force or torque required to fix the positions or rotation are reported.
      
   .. note::
      This plugin is inactive if postprocess is disabled
   
   Additional attributes:
   
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+
| Attribute       | Type     | Default | Remarks                                                                                                    |
+=================+==========+=========+============================================================================================================+
| ov_name         | string   | ""      | Name of the OV                                                                                             |
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+
| dump_every      | integer  | 1000    | Write every this many time-steps                                                                           |
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+
| path            | string   | "pos/"  | The filenames with force will look like this: <path>/<ov_name>_NNNNN.txt                                   |
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+
| pin_translation | integer3 | (0,0,0) | 0 means that motion along the corresponding axis is unrestricted, 1 means fixed position wrt to the axis   |
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+
| pin_rotation    | integer3 | (0,0,0) | 0 means that rotation along the corresponding axis is unrestricted, 1 means fixed rotation wrt to the axis |
+-----------------+----------+---------+------------------------------------------------------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="pin_object" name="pin"
         ov_name="blob"
         dump_every="10000"
         path="pinning_force/"
         pin_translation="0 1 1" />
         
               
* **Imposing velocity in an area**
   This plugin will add velocity to all the particles of the target PV in the specified area (rectangle) such that the average velocity equals to desired.
   
   Additional attributes:
   
+-----------------+----------+---------+--------------------------------------------------+
| Attribute       | Type     | Default | Remarks                                          |
+=================+==========+=========+==================================================+
| pv_name         | string   | ""      | Name of the PV                                   |
+-----------------+----------+---------+--------------------------------------------------+
| every           | integer  | 5       | Correct the velocity every this many time-steps  |
+-----------------+----------+---------+--------------------------------------------------+
| low             | float3   | (0,0,0) | Lower corner of the affected rectangular volume  |
+-----------------+----------+---------+--------------------------------------------------+
| high            | integer3 | (0,0,0) | Higher corner of the affected rectangular volume |
+-----------------+----------+---------+--------------------------------------------------+
| target_velocity | integer3 | (0,0,0) | Target velocity                                  |
+-----------------+----------+---------+--------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <plugin type="impose_velocity" name="push" pv_name="dpd" every="10" low="10 20 30"  high="15 25 50"  target_velocity= "-1 2 3" />

         