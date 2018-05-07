.. _user-bouncers:

Object bouncers
###############

Bouncers prevent liquid particles crossing boundaries of objects (maintaining no-through boundary conditions).
The idea of the bouncers is to move the particles that crossed the object boundary after integration step back to the correct side.
Particles are moved such that they appear very close (about :math:`10^{-4}` units away from the boundary).
Assuming that the objects never come too close to each other or the walls,
that approach ensures that recovered particles will not penetrate into a different object or wall.
In practice maintaining separation of at least :math:`10^{-3}` units between walls and objects is sufficient.
Note that particle velocities are also altered, which means that objects experience extra force from the collisions.
This force is saved for the next timestep.

Syntax
******

.. role:: xml(code)
   :language: xml

.. code-block:: xml

   <object_bouncer attributes="..." >
      <apply_to pv="name" />
   </object_bouncer>

The bouncer defined by the :xml:`bouncer` node will be applied to all the Particle Vectors that are listed in :xml:`apply_to` children of that node.

Common attributes
*****************

+-----------+--------+---------+----------------------------------------------------------------------+
| Attribute | Type   | Default | Remarks                                                              |
+===========+========+=========+======================================================================+
| type      | string | ""      | Type of the bouncer, see below for the                               |
|           |        |         | list of available types                                              |
+-----------+--------+---------+----------------------------------------------------------------------+
| name      | string | ""      | Name of the created bouncer                                          |
+-----------+--------+---------+----------------------------------------------------------------------+
| ov        | string | ""      | Name of the object vector that the particles will be bounced against |
+-----------+--------+---------+----------------------------------------------------------------------+

Available Bouncers
******************

* **Mesh bouncer**

   Type: *from_mesh*
   
   This bouncer will use the triangular mesh associated with objects to detect boundary crossings.
   Therefore it can only be created for Membrane and Rigid Object types of object vectors.
   Due to numerical precision, about :math:`1` of :math:`10^5 - 10^6` mesh crossings will not be detected, therefore it is advised to use that bouncer in
   conjunction with correction option provided by the Object Belonging Checker, see :ref:`user-belongers`.
   
   .. note:
      In order to prevent numerical instabilities in case of light membrane particles,
      the new velocity of the bounced particles will be a random vector drawn from the Maxwell distibution of given temperature
      and added to the velocity of the mesh triangle at the collision point.
   
   Additional attributes:
   
   +-----------+-------+---------+-------------------------------------------------------------------+
   | Attribute | Type  | Default | Remarks                                                           |
   +===========+=======+=========+===================================================================+
   | kbt       | float | 0.5     | Maxwell distribution temperature defining post-collision velocity |
   +-----------+-------+---------+-------------------------------------------------------------------+

   **Example**
   
   .. code-block:: xml
   
      <object_bouncer name="membrane" ov="membrane" type="from_mesh">
         <apply_to pv="outside" />
         <apply_to pv="inside" />
      </object_bouncer>
   
   
* **Rigid ellipsoid bouncer**

   Type: *from_ellipsoids*
   
   This bouncer will use the analytical ellipsoid representation to perform the bounce.
   No additional correction from the Object Belonging Checker is usually required.
   The velocity of the particles bounced from the ellipsoid is reversed with respect to the boundary velocity at the contact point.
   
   Additional attributes:
  
   None

   **Example**
   
   .. code-block:: xml
   
      <object_bouncer name="ell_bounce" ov="ellipsoid" type="from_ellipsoids" >
         <apply_to pv="dpd" />
      </object_bouncer>
   
   
   
   
   
   
   
   
   
   