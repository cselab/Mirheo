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
   
   Pairwise interaction with conservative part and dissipative + random part acting a thermostat, see https://aip.scitation.org/doi/abs/10.1063/1.474784
   
   .. math::
   
      \mathbf{F}_{ij} &= \mathbf{F}^C(\mathbf{r}_{ij}) + \mathbf{F}^D(\mathbf{r}_{ij}, \mathbf{u}_{ij}) + \mathbf{F}^R(\mathbf{r}_{ij}) \\
      \mathbf{F}^C(\mathbf{r}) &= \begin{cases} a(1-\frac{r}{r_c}) \mathbf{\hat r}, & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
      \mathbf{F}^D(\mathbf{r}, \mathbf{u}) &= \gamma w^2(\frac{r}{r_c}) (\mathbf{r} \cdot \mathbf{u}) \mathbf{\hat r} \\
      \mathbf{F}^R(\mathbf{r}) &= \sigma w(\frac{r}{r_c}) \, \theta \sqrt{\Delta t} \, \mathbf{\hat r}
   
   where bold symbol means a vector, its regular counterpart means vector length: 
   :math:`x = \left\lVert \mathbf{x} \right\rVert`, hat-ed symbol is the normalized vector:
   :math:`\mathbf{\hat x} = \mathbf{x} / \left\lVert \mathbf{x} \right\rVert`. Moreover, :math:`\theta` is the random variable with zero mean
   and unit variance, that is distributed independently of the interacting pair *i*-*j*, dissipation and random forces 
   are related by the fluctuation-dissipation theorem: :math:`\sigma^2 = 2 \gamma \, k_B T`; and :math:`w(r)` is the weight function
   that we define as follows:
   
   .. math::
      
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
   
      <interaction type="dpd" name="dpd_int" a="25" gamma="40" kbt="0.5" dt="0.001" rc="1.0" power="0.5">
         <apply_to pv1="dpd"  pv2="dpd" />
         <apply_to pv1="wall" pv2="dpd" gamma="120" power="1.0" />    
      </interaction>

* **Pairwise Lennard-Jones interaction**

   Type: *lj*
   
   Pairwise interaction according to the classical Lennard-Jones potential `http://rspa.royalsocietypublishing.org/content/106/738/463`
   The force however is truncated such that it is *always repulsive*.
   
   
   .. math::
   
      \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
   
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
   
      <interaction type="lj" name="lj_int" epsilon="0.1" sigma="0.5" rc="1.0" >
         <apply_to pv1="object" pv2="wall" />
      </interaction>
      
      
      
* **Pairwise Lennard-Jones interaction object-aware**

   Type: *lj_object*
   
   Same as regular LJ interaction, but the particles belonging to the same object in an object vector do not interact with each other.
   That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 

   **Example**
   
   .. code-block:: xml
   
      <interaction type="lj_object" name="lj_obj_int" epsilon="0.1" sigma="0.5" rc="1.0" >
         <apply_to pv1="membrane" pv2="membrane" />
      </interaction>


* **Membrane**

   Type: *membrane*
   
   Mesh-based forces acting on a membrane according to the model in LINK
   
   Additional attributes:
   
+-----------+--------+---------+----------------------------------------------------+
| Attribute | Type   | Default | Remarks                                            |
+===========+========+=========+====================================================+
| preset    | string | ""      | Set the parameters to predifined. Accepted values: |
|           |        |         | * "lina":                                          |
+-----------+--------+---------+----------------------------------------------------+
| x0        | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| p         | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| ka        | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| kb        | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| kd        | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| kv        | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| gammaC    | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| gammaT    | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| kbT       | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| mpow      | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| theta     | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| area      | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
| volume    | float  | 0.0     |                                                    |
+-----------+--------+---------+----------------------------------------------------+
                                  
    **Example**                   
                                  
   .. code-block:: xml            
                                  
      <particle_vector type="membrane" name="rbcs" mass="1.0" particles_per_obj="498" mesh_filename="rbc_mesh.off"  >
         <generate type="restart" path="restart/" />
      </particle_vector>
      

      
