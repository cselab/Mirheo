.. _user-ic:

Initial conditions
##################

Initial conditions (IC) have to be defined for every Particle Vector

Syntax
******

.. role:: xml(code)
   :language: xml


.. code-block:: xml

   <particle_vector>
      <generate attributes="..."/>
   </particle_vector>

.. note::
   :xml:`particle_vector` node defines the type of Particle Vector to which the IC will be applied, see :ref:`user-ic`

Common attributes
*****************

+-----------+--------+---------+-----------------------------------+
| Attribute | Type   | Default | Remarks                           |
+===========+========+=========+===================================+
| type      | string | ""      | Type of the IC, see below for the |
|           |        |         | list of available types           |
+-----------+--------+---------+-----------------------------------+
                      


Available Initial Conditions
****************************

* **Uniform Random**

   Type: *uniform*
   
   The particles will be generated with the desired number density uniformly at random in all the domain.
   These IC may be used with any Particle Vector, but only make sense for regular PV.
   
+-----------+-------+---------+-----------------------+
| Attribute | Type  | Default | Remarks               |
+===========+=======+=========+=======================+
| density   | float | 1.0     | Target number density |
+-----------+-------+---------+-----------------------+
                    
   **Example**      
   
   .. code-block:: xml
   
      <particle_vector type="regular" name="dpd" mass="1"  >
         <generate type="uniform" density="8" />
      </particle_vector>

* **Restart**

   Type: *restart*
   
   Read the state (particle coordinates and velocities, other relevant data for objects **not implemented yet**)
   
   Additional attributes:
   
   +-----------+--------+------------+---------------------------------------------------------------------------------------------------+
   | Attribute | Type   | Default    | Remarks                                                                                           |
   +===========+========+============+===================================================================================================+
   | path      | string | "restart/" | Folder where the restart files reside. The exact filename will be like this: <path>/<PV name>.chk |
   +-----------+--------+------------+---------------------------------------------------------------------------------------------------+
   
    **Example**
   
   .. code-block:: xml
   
      <particle_vector type="membrane" name="rbcs" mass="1.0" particles_per_obj="498" mesh_filename="rbc_mesh.off"  >
         <generate type="restart" path="restart/" />
      </particle_vector>
      
* **Rigid objects from template**

   Type: *read_rigid*
   
   Can only be used with Rigid Object Vector or Rigid Ellipsoid, see :ref:`user-ic`. These IC will initialize the particles of each object
   according to the template .xyz file and then the objects will be translated/rotated according to the file initial conditions file.
   
   Additional attributes:
   
   +--------------+---------+--------------+-------------------------------------------------------------------------------------------+
   | Attribute    | Default | Type         | Remarks                                                                                   |
   +==============+=========+==============+===========================================================================================+
   | ic_filename  | string  | "objects.ic" | Text file describing location and rotation of the created objects.                        |
   |              |         |              | One line in the file corresponds to one object created.                                   |
   |              |         |              | Format of the line: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where             |
   |              |         |              | *com* is the center of mass of the object, *q* is the quaternion of its rotation,         |
   |              |         |              | not necessarily normalized                                                                |
   +--------------+---------+--------------+-------------------------------------------------------------------------------------------+
   | xyz_filename | string  | "object.xyz" | Template that describes the positions of the body particles before translation or         |
   |              |         |              | rotation is applied. Standard .xyz file format is used with first line being              |
   |              |         |              | the number of particles, second comment, third and onwards - particle coordinates.        |
   |              |         |              | The number of particles in the file should be the same as in the :xml:`particles_per_obj` |
   |              |         |              | attribute of the corresponding PV                                                         |
   +--------------+---------+--------------+-------------------------------------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <particle_vector type="rigid_objects" name="blob" mass="1.0" particles_per_obj="4242" moment_of_inertia="67300 45610 34300" mesh_filename="blob.off" >
          <generate type="read_rigid" ic_filename="blob.ic" xyz_filename="blob.xyz"/>
      </particle_vector>

   
   
* **Membranes**

   Type: *read_membranes*
   
   Can only be used with Membrane Object Vector, see :ref:`user-ic`. These IC will initialize the particles of each object
   according to the mesh associated with Membrane, and then the objects will be translated/rotated according to the file initial conditions file.
   
   Additional attributes:
   
   +--------------+---------+----------------+---------------------------------------------------------------------------------------------------+
   | Attribute    | Default | Type           | Remarks                                                                                           |
   +==============+=========+================+===================================================================================================+
   | ic_filename  | string  | "membranes.ic" | Text file describing location and rotation of the created objects.                                |
   |              |         |                | One line in the file corresponds to one object created.                                           |
   |              |         |                | Format of the line: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where                     |
   |              |         |                | *com* is the center of mass of the object, *q* is the quaternion of its rotation,                 |
   |              |         |                | not necessarily normalized                                                                        |
   +--------------+---------+----------------+---------------------------------------------------------------------------------------------------+
   | global_scale | float   | 1.0            | All the membranes will be scaled by that value. Useful to implement membranes growth so that they |
   |              |         |                | so that they can fill the space with high volume fraction                                         |
   +--------------+---------+----------------+---------------------------------------------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <particle_vector type="membrane" name="rbcs" mass="1.0" particles_per_obj="498" mesh_filename="rbc_mesh.off" >
          <generate type="read_rbcs" ic_filename="rbcs.ic" global_scale="0.5" />
      </particle_vector>
