.. _dev-packers:

Packers
=======

Packers are used to store data of a set of registered channels from a :any:`mirheo::DataManager` into a single buffer and vice-versa.
They are used to redistribute and exchange data accross neighbouring ranks efficiently.
This allows to send single MPI messages instead of one message per channel.

Generic Packer
--------------

This is the base packer class.
All packers contain generic packers that are used to pack different kind of data (such as particle or object data).

.. doxygenstruct:: mirheo::GenericPackerHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::GenericPacker
   :project: mirheo
   :members:


Particles Packer
----------------

.. doxygenstruct:: mirheo::ParticlePackerHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticlePacker
   :project: mirheo
   :members:


Objects Packer
--------------

.. doxygenstruct:: mirheo::ObjectPackerHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjectPacker
   :project: mirheo
   :members:


Rods Packer
-----------

.. doxygenstruct:: mirheo::RodPackerHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::RodPacker
   :project: mirheo
   :members:
