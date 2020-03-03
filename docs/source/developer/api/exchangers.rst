.. _dev-exchangers:

Exchangers
==========

A set of classes responsible to:

- exchange ghost paticles between neighbouring ranks
- redistribite particles accross ranks

The implementation is split into two parts:

- :ref:`exchanger classes<dev-exchangers-exchangers>`, that are responsible to pack and unpack the data from :any:`mirheo::ParticleVector` to buffers (see also :ref:`dev-packers`).
- :ref:`communication engines<dev-exchangers-engines>`, that communicate the buffers created by the exchangers between ranks. The user must instantiate one engine per exchanger.

.. _dev-exchangers-exchangers:

Exchanger classes
-----------------

Different kind of exchange are implemented in Mirheo:

- Redistribution: the data is migrated from one rank to another
  (see :any:`mirheo::ParticleRedistributor` and :any:`mirheo::ObjectRedistributor`)
- Ghost particles: the data is copied from one rank to possibly multiple ones
  (see :any:`mirheo::ParticleHaloExchanger`, :any:`mirheo::ObjectHaloExchanger` and :any:`mirheo::ObjectExtraExchanger`)
- Reverse exchange: data is copied from possibly multiple ranks to another.
  This can be used to gather e.g. the forces computed on ghost particles,
  and therefore is related to the ghost particles exchangers.
  (see :any:`mirheo::ObjectReverseExchanger`)

In general, the process consists in:

#. Create a map from particle/object to buffer(s) (this step might be unnecessary for e.g. :any:`mirheo::ObjectExtraExchanger` and :any:`mirheo::ObjectReverseExchanger`)
#. Pack the data into the send buffers according to the map
#. The :ref:`communication engines<dev-exchangers-engines>` communicate the data to the recv buffers (not the exchangers job)
#. Unpack the data from recv buffers to a local container.

Interface
^^^^^^^^^

.. doxygenclass:: mirheo::Exchanger
   :project: mirheo
   :members:


Derived classes
^^^^^^^^^^^^^^^

.. doxygenclass:: mirheo::ParticleRedistributor
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjectRedistributor
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleHaloExchanger
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjectHaloExchanger
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjectExtraExchanger
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjectReverseExchanger
   :project: mirheo
   :members:

Exchange Entity
^^^^^^^^^^^^^^^

Helper classes responsible to hold the buffers of the packed data to be communicated.

.. doxygenstruct:: mirheo::BufferOffsetsSizesWrap
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::BufferInfos
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ExchangeEntity
   :project: mirheo
   :members:



.. _dev-exchangers-engines:

Communication engines
---------------------

Interface
^^^^^^^^^

.. doxygenclass:: mirheo::ExchangeEngine
   :project: mirheo
   :members:

Derived classes
^^^^^^^^^^^^^^^

.. doxygenclass:: mirheo::MPIExchangeEngine
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::SingleNodeExchangeEngine
   :project: mirheo
   :members:

