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

TODO

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

