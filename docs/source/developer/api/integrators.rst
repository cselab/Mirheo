.. _dev-integrators:

Integrators
===========

See also :ref:`the user interface <user-integrators>`.

Base class
----------

.. doxygenclass:: mirheo::Integrator
   :project: mirheo
   :members:

Derived classes
---------------

.. doxygenclass:: mirheo::IntegratorConstOmega
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::IntegratorOscillate
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::IntegratorVVRigid
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::IntegratorSubStep
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::IntegratorTranslate
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::IntegratorVV
   :project: mirheo
   :members:


Forcing terms
-------------

The forcing terms must follow the following interface:

.. doxygenclass:: mirheo::ForcingTerm
   :project: mirheo
   :members:

Currently implemented forcing terms:

.. doxygenclass:: mirheo::ForcingTermNone
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ForcingTermConstDP
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ForcingTermPeriodicPoiseuille
   :project: mirheo
   :members:


