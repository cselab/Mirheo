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
| name      | string | ""      | Name of the created interaction            |
+-----------+--------+---------+--------------------------------------------+
| rc        | float  | 1.0     | Cut-off radius of the pairwise interaction |
+-----------+--------+---------+--------------------------------------------+

Available Particle Vectors
**************************

* **Pairwise DPD interaction**

   Type: *dpd*
   
   Pairwise interaction with conservative part and dissipative + random part acting a thermostat, see `https://aip.scitation.org/doi/abs/10.1063/1.474784`
   
   .. math::
   
      \mathbf{F}_{ij} &= \mathbf{F}^C(\mathbf{r}_{ij}) + \mathbf{F}^D(\mathbf{r}_{ij}, \mathbf{u}_{ij}) + \mathbf{F}^R(\mathbf{r}_{ij}) \\
      \mathbf{F}^C(\mathbf{r}) &= \begin{cases} a(1-\frac{r}{r_c}) \mathbf{\hat r}, & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
      \mathbf{F}^D(\mathbf{r}, \mathbf{u}) &= \gamma w^2(\frac{r}{r_c}) (\mathbf{r} \cdot \mathbf{u}) \mathbf{\hat r} \\
      \mathbf{F}^R(\mathbf{r}) &= \sigma w(\frac{r}{r_c}) \, \theta \sqrt{\Delta t} \, \mathbf{\hat r}
   
   where bold symbol means a vector, its regular counterpart means vector length: 
   :math:`x = \left\lVert \mathbf{x} \right\rVert`, hat-ed symbol is the normalized vector:
   :math:`\mathbf{\hat x} = \mathbf{x} / \left\lVert \mathbf{x} \right\rVert`. Moreover, :math:`\theta` is the random variable with zero mean
   and unit variance, that is distributed independently of the interacting pair *i*-*j*, dissipation and random forces 
   are related by the fluctuation-dissipation theorem: :math:`\sigma^2 = 2 \gamma k_B T`; and :math:`w(r)` is the weight function
   that we define as follows:
   
   .. math:
      
      w(r) = \begin{cases} (1-r)^{p}, & r < 1 \\ 0, & r \geqslant 1 \end{cases}
      
   Additional attributes:
   
   +-----------+-------+---------+------------------------+
   | Attribute | Type  | Default | Remarks                |
   +===========+=======+=========+========================+
   | a         | float | 50.0    |                        |
   +-----------+-------+---------+------------------------+
   | gamma     | float | 20.0    |                        |
   +-----------+-------+---------+------------------------+
   | kbT       | float | 1.0     |                        |
   +-----------+-------+---------+------------------------+
   | dt        | float | 0.01    |                        |
   +-----------+-------+---------+------------------------+
   | power     | float | 1.0     | *p* in weight function |
   +-----------+-------+---------+------------------------+


   **Example**
   
   
   .. code-block:: xml
   
      <particle_vector type="regular" name="dpd" mass="1"  >
         <generate type="uniform" density="8" />
      </particle_vector>

* **Pairwise Lennard-Jones interaction**

   Type: *lj*
   
   Pairwise interaction according to the classical Lennard-Jones potential `http://rspa.royalsocietypublishing.org/content/106/738/463`
   The force however is truncated such that it is *always repulsive*.
   
   
   .. math::
   
      \mathbf{F}_{ij} &= \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
      
   Additional attributes:
   
   +-----------+-------+---------+---------+
   | Attribute | Type  | Default | Remarks |
   +===========+=======+=========+=========+
   | sigma     | float | 0.5     |         |
   +-----------+-------+---------+---------+
   | epsilon   | float | 10.0    |         |
   +-----------+-------+---------+---------+


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
      

      
