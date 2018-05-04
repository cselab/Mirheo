.. _user-integrators:

Integrators
###########

Integrators are used to advance particle coordinates and velocities in time according to forces acting on them.

.. note:
   For now all the integrators in the simulation must use the same timestep

Syntax
******

.. role:: xml(code)
   :language: xml

.. code-block:: xml

   <integrator attributes="..." >
      <apply_to pv="name1" />
   </integrator >

The integrator defined by the :xml:`integrator` node will be applied to all the Particle Vectors that are listed in :xml:`apply_to` children of that node.

Common attributes
*****************

+-----------+--------+---------+-------------------------------------------+
| Attribute | Type   | Default | Remarks                                   |
+===========+========+=========+===========================================+
| type      | string | ""      | Type of the integrator, see below for the |
|           |        |         | list of available types                   |
+-----------+--------+---------+-------------------------------------------+
| name      | string | ""      | Name of the created interaction           |
+-----------+--------+---------+-------------------------------------------+
| dt        | float  | 0.01    | Time-step                                 |
+-----------+--------+---------+-------------------------------------------+

Available Integrators
*********************

* **Velocity-Verlet**

   Type: *vv*
   
   Classical Velocity-Verlet integrator with fused steps for coordinates and velocities.
   The velocities are shifted with respect to the coordinates by one half of the time-step
   
   .. math::
   
      \mathbf{a}^{n} &= \frac{1}{m} \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) \\
      \mathbf{v}^{n+1/2} &= \mathbf{v}^{n-1/2} + \mathbf{a}^n \Delta t \\
      \mathbf{x}^{n+1} &= \mathbf{x}^{n} + \mathbf{v}^{n+1/2} \Delta t 
   
   where bold symbol means a vector, :math:`m` is a particle mass, and superscripts denote the time: :math:`\mathbf{x}^{k} = \mathbf{x}(k \, \Delta t)`
   
   No additional attributes


   **Example**
   
   
   .. code-block:: xml
   
      <integrator type="vv" name="integrate" dt="0.001">
         <apply_to pv="dpd" />
         <apply_to pv="membrane" />
      </integrator>

* **Velocity-Verlet with constant pressure term**

   Type: *vv_const_dp*
   
   Same as regular Velocity-Verlet, but the forces on all the particles are modified with the constant pressure term:
   
   .. math::
   
      \mathbf{a}^{n} &= \frac{1}{m} \left( \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) + \mathbf{F}_{extra} \right) \\
   
   
   Additional attributes:
   
   +-------------+--------+-----------+----------------------------+
   | Attribute   | Type   | Default   | Remarks                    |
   +=============+========+===========+============================+
   | extra_force | float3 | (0, 0, 0) | :math:`\mathbf{F}_{extra}` |
   +-------------+--------+-----------+----------------------------+

   **Example**
   
   
   .. code-block:: xml
   
      <integrator type="vv_const_dp" name="push" extra_force="0.1 0 0" dt="0.005">
         <apply_to pv="liquid" />
      </integrator>
      
      
* **Velocity-Verlet with periodic Poiseuille term**

   Type: *vv_periodic_poiseuille*
   
   Same as regular Velocity-Verlet, but the forces on all the particles are modified with periodic Poiseuille term.
   This means that all the particles in half domain along certain axis (Ox, Oy or Oz) are pushed with force
   :math:`F_{Poiseuille}` parallel to Oy, Oz or Ox correspondingly, and the particles in another half of the domain are pushed in the same direction
   with force :math:`-F_{Poiseuille}`    
   
   Additional attributes:
   
   +-----------+--------+---------+-------------------------------------------------------------------------+
   | Attribute | Type   | Default | Remarks                                                                 |
   +===========+========+=========+=========================================================================+
   | direction | string | "x"     | Valid values: "x", "y", "z". Defines the direction of the pushing force |
   +-----------+--------+---------+-------------------------------------------------------------------------+
   | force     | float  | 0.1     | Force magnitude, :math:`F_{Poiseuille}`                                 |
   +-----------+--------+---------+-------------------------------------------------------------------------+

   **Example**
   
   
   .. code-block:: xml
   
      <integrator type="vv_periodic_poiseuille" name="poiseuille" direction="x" force="0.1" dt="0.0025">
         <apply_to pv="liquid" />
      </integrator>

* **Rigid body Velocity-Verlet integration**

   Type: *rigid_vv*
   
   Integrate the position and rotation (in terms of quaternions) of the rigid bodies as per Velocity-Verlet scheme.
   Can only applied to Rigid Object Vector or Rigid Ellipsoid Object Vector.
   
   No additional attributes
                                  
   **Example**                   
                               
   .. code-block:: xml            
                                  
      <integrator type="rigid_vv" name="rigid" dt="0.001">
         <apply_to pv="rigid_bodies" />
      </integrator>
      
* **Translate with constant velocity**

   Type: *translate*
   
   Translate particles with a constant velocity :math:`\mathbf{U}` regardless forces acting on them.
   
   +-----------+--------+---------+--------------------+
   | Attribute | Type   | Default | Remarks            |
   +===========+========+=========+====================+
   | velocity  | float3 | (0,0,0) | :math:`\mathbf{U}` |
   +-----------+--------+---------+--------------------+
                                  
   **Example**                   
                               
   .. code-block:: xml            
                                  
      <integrator type="translate" name="move" velocity="0.1 0.2 0.3" dt="0.001">
         <apply_to pv="pv_name" />
      </integrator>
      
      
* **Rotate with constant angular velocity**

   Type: *const_omega*
   
   Rotate particles around the specified point in space with a constant angular velocity :math:`\mathbf{\Omega}`
   
   +-----------+--------+---------+-------------------------+
   | Attribute | Type   | Default | Remarks                 |
   +===========+========+=========+=========================+
   | center    | float3 | (0,0,0) |                         |
   +-----------+--------+---------+-------------------------+
   | omega     | float3 | (0,0,0) | :math:`\mathbf{\Omega}` |
   +-----------+--------+---------+-------------------------+
                                  
   **Example**                   
                               
   .. code-block:: xml            
                                  
      <integrator type="const_omega" name="rotate" center="10 10 10" omega="0.5 0 0" dt="0.001">
         <apply_to pv="cylinder" />
      </integrator>
      
      
