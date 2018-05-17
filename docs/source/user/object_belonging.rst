.. _user-belongers:

Object belonging checkers
#########################


Object belonging checkers serve two purpooses.
Firstly, they are used to split a particle vector into two disjointed parts (probably forming a new Particle Vector):
the particles that are *inside* any object of the given Object Vector and the particles that are *outside*.
Secondly, they are used to maintain the mentioned *inside*-*outside* property of the particles in the resulting Particle Vectors.

Syntax
******

.. role:: xml(code)
   :language: xml

.. code-block:: xml

   <object_belonging_checker attributes="..." >
      <apply_to pv="name" attributes="..." />
   </object_belonging_checker>

The bouncer defined by the :xml:`<object_belonging_checker>` node will be applied to all the Particle Vectors that are listed in :xml:`<apply_to>` children of that node.

Common attributes
*****************

+---------------+--------+---------+-------------------------------------------------------------------------------------------+
| Attribute     | Type   | Default | Remarks                                                                                   |
+===============+========+=========+===========================================================================================+
| type          | string | ""      | Type of the object belonging checker, see below for the                                   |
|               |        |         | list of available types                                                                   |
+---------------+--------+---------+-------------------------------------------------------------------------------------------+
| name          | string | ""      | Name of the created object belonging checker                                              |
+---------------+--------+---------+-------------------------------------------------------------------------------------------+
| object_vector | string | ""      | Name of the object vector that will be used to determine *inside* and *outside* particles |
+---------------+--------+---------+-------------------------------------------------------------------------------------------+


Attributes of the  :xml:`apply_to` node are:

+---------------+--------+---------+----------------------------------------------------------------------------------------------+
| Attribute     | Type   | Default | Remarks                                                                                      |
+===============+========+=========+==============================================================================================+
| pv            | string | ""      | Name of the particle vector that will be split (source PV)                                   |
+---------------+--------+---------+----------------------------------------------------------------------------------------------+
| inside        | string | ""      | Name of the particle vector that will contain the *inner* particles.                         |
|               |        |         | The name may be one of three varians:                                                        |
|               |        |         | * same as the source PV: the source PV will be modified to only contain *inner* particles    |
|               |        |         | * new PV name: a new PV will be created that will have the *inner* particles                 |
|               |        |         | * "none": all the *inner* particles will be discarded from the simulation. That option       |
|               |        |         | is useful to remove liquid particles from inside the rigid objects                           |
+---------------+--------+---------+----------------------------------------------------------------------------------------------+
| outside       | string | ""      | Name of the particle vector that will contain the *outer* particles.                         |
|               |        |         | Same rules as for the :xml:`inside` atttibute apply                                          |
+---------------+--------+---------+----------------------------------------------------------------------------------------------+
| correct_every | int    | 0       | If greater than zero, perform correction every this many time-steps.                         |
|               |        |         | Correction will move e.g. *inner* particles of the :xml:`outside` PV to the :xml:`inside` PV |
|               |        |         | and viceversa. If one of the PVs was defined as "none", the wrong particles will be removed. |
+---------------+--------+---------+----------------------------------------------------------------------------------------------+


Available Belonging Checkers
****************************

* **Mesh checker**

   Type: *mesh*
   
   This checker will use the triangular mesh associated with objects to detect *inside*-*outside* status.
   
   .. note:
      Checking if particles are inside or outside the mesh is a computationally expensive task,
      so it's best to perform checks at most every 1'000 - 10'000 time-steps.
   
   Additional attributes:
   
   None

   **Example**
   
   .. code-block:: xml
   
      <!-- At the beginning of the simulation "dpd" PV (usually created randomly with uniform density)
           will be stripped off of all the particles inside the membranes definde by OV "rbcs".
           Those inner particles will form a new PV "inner" -->
            
      <object_belonging_checker name="membrane_checker" type="mesh" object_vector="rbcs"  >
         <apply_to pv="dpd" inside="ins" outside="dpd" check_every="10000" />
      </object_belonging_checker>
   
   
* **Ellipsoid checker**

   Type: *analytical_ellipsoid*
   
   This checker will use the analytical representation of the ellipsoid to detect *inside*-*outside* status.
   
   Additional attributes:
  
   None

   **Example**
   
   .. code-block:: xml
   
      <object_belonging_checker name="ell_checker" type="analytical_ellipsoid" object_vector="spheres"  >
         <apply_to pv="liquid" inside="none" outside="liquid" check_every="50000" />
      </object_belonging_checker>
   
   
   



