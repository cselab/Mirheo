.. _user-interactions:

Interactions
############

Interactions are used to calculate forces on individual particles due to their neighbours.
Pairwise short-range interactions are supported now, and membrane forces

Syntax
******

.. role:: xml(code)
   :language: xml

.. code-block:: xml

   <interaction attributes="..." >
      <apply_to pv1="name1" pv2="name2" attributes="..."/>
   </interaction >

Forces between two Particle Vectors (they can be the same) *name1* and *name2* will be computed according to the defined interaction.
Interaction parameters are specified as the attributes of the node :xml:`interaction`, while pairwise interaction may override those
basic parameters on per-pair basis by defining relevant parameters in the attributes of the corresponding :xml:`apply_to` node.

Common attributes
*****************

+-----------+--------+---------+--------------------------------------------+
| Attribute | Type   | Default | Remarks                                    |
+===========+========+=========+============================================+
| type      | string | ""      | Type of the interaction, see below for the |
|           |        |         | list of available types                    |
+-----------+--------+---------+--------------------------------------------+
| rc        | float  | 1.0     | Cut-off radius of the pairwise interaction |
+-----------+--------+---------+--------------------------------------------+

Available Particle Vectors
**************************

* **Pairwise DPD interaction**

   Type: *dpd*
   
   Pairwise interaction with conservative part and dissipative + random part acting a thermostat, see `https://aip.scitation.org/doi/abs/10.1063/1.474784`
   
   .. math::
   
      \mathbb{F}_{ij} &= \mathbb{F}^C(\mathbb{r}_{ij}) + \mathbb{F}^D(\mathbb{r}_{ij}, \mathbb{u}_{ij}) + \mathbb{F^R(\mathbb{r}_{ij}) \\
      \mathbb{F}^C(r) &= \begin{cases} a(1-\frac{r}{r_c}) \mathbb{r}, & r < r_c; \\ 0, & r \geqslant r_c \end{cases} \\
      \mathbb{F}^D(r, u) &= \gamma w^2(\frac{r}{r_c}) (\mathbb{r} \dot \mathbb{u}) \mathbb{r} \\
      \mathbb{F}^D(r, u) &= \gamma w^2(\frac{r}{r_c}) (\mathbb{r} \dot \mathbb{u}) \mathbb{r}


   **Example**
   
   
   .. code-block:: xml
   
      <particle_vector type="regular" name="dpd" mass="1"  >
         <generate type="uniform" density="8" />
      </particle_vector>

* **Membrane**

   Type: *membrane*
   
   Membrane is an Object Vector representing cell membranes.
   It must have a triangular mesh associated with it such that each particle is mapped directly onto single mesh vertex.
   
   Additional attributes:
   
   +-------------------+---------+----------+----------------------------------------------+
   | Attribute         | Type    | Default  | Remarks                                      |
   +===================+=========+==========+==============================================+
   | particles_per_obj | integer | 1        | Number of the particles making up one cell   |
   +-------------------+---------+----------+----------------------------------------------+
   | mesh_filename     | string  |          | Path to the .OFF mesh file, see `OFF mesh`.  |
   |                   |         | mesh.off | The number of vertices of the mesh should be |
   |                   |         |          | equal to :xml:`particles_per_obj`.           |
   +-------------------+---------+----------+----------------------------------------------+
                                  
    **Example**                   
                                  
   .. code-block:: xml            
                                  
      <particle_vector type="membrane" name="rbcs" mass="1.0" particles_per_obj="498" mesh_filename="rbc_mesh.off"  >
         <generate type="restart" path="restart/" />
      </particle_vector>
      
* **Rigid object**

   Type: *rigid_objects*
   
   Rigid Object is an Object Vector representing objects that move as rigid bodies, with no relative displacement against each other in an object.
   It must have a triangular mesh associated with it that defines the shape of the object.
   
   Additional attributes:
   
   +-------------------+---------+-----------+----------------------------------------------------------------------------------------------+
   | Attribute         | Type    | Default   | Remarks                                                                                      |
   +===================+=========+===========+==============================================================================================+
   | particles_per_obj | integer | 1         | Number of the particles making up one cell                                                   |
   +-------------------+---------+-----------+----------------------------------------------------------------------------------------------+
   | mesh_filename     | string  |           | Path to the .OFF mesh file, see `OFF mesh`.                                                  |
   |                   |         | mesh.off  | The number of vertices of the mesh should be                                                 |
   |                   |         |           | equal to :xml:`particles_per_obj`.                                                           |
   +-------------------+---------+-----------+----------------------------------------------------------------------------------------------+
   | moment_of_inertia | float3  | (1, 1, 1) | Moment of inertia of the body in its principal axes                                          |
   |                   |         |           | The principal axes of the mesh are assumed to be aligned with the default global *OXYZ* axes |
   +-------------------+---------+-----------+----------------------------------------------------------------------------------------------+
   
   **Example**
   
   .. code-block:: xml
   
      <particle_vector type="rigid_objects" name="blob" mass="1.0" particles_per_obj="4242" moment_of_inertia="67300 45610 34300" mesh_filename="blob.off" >
          <generate type="read_rigid" ic_filename="blob.ic" xyz_filename="blob.xyz"/>
      </particle_vector>

   
* **Rigid ellipsoid**

   Type: *rigid_ellipsoids*
   
   Rigid Ellipsoid is the same as the Rigid Object except that it can only represent ellipsoidal shapes.
   The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
   
   Additional attributes:
   
   +-------------------+---------+-----------+--------------------------------------------+
   | Attribute         | Type    | Default   | Remarks                                    |
   +===================+=========+===========+============================================+
   | particles_per_obj | integer | 1         | Number of the particles making up one cell |
   +-------------------+---------+-----------+--------------------------------------------+
   | axes              | float3  | (1, 1, 1) | Ellipsoid principal semi-axes              |
   +-------------------+---------+-----------+--------------------------------------------+
   
   **Example**                   
   
   .. code-block:: xml
   
      <particle_vector type="rigid_ellipsoids" name="sphere" mass="1.847724" particles_per_obj="2267" axes="5 5 5" >
           <generate type="read_rigid" ic_filename="sphere.ic" xyz_filename="sphere.xyz" />
      </particle_vector>
      

      
