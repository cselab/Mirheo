.. _dev-config:

Config
======

:any:`mirheo::ConfigValue` is a JSON-like type used to represent Mirheo's internal setup and state, for the purpose of :ref:`snapshotting <dev-snapshots>`.
See also the snapshotting :ref:`API reference <dev-api-snapshot>`.

API
---

.. doxygenfunction:: mirheo::configFromJSONFile
   :project: mirheo

.. doxygenfunction:: mirheo::configFromJSON
   :project: mirheo

.. doxygenfunction:: mirheo::parseNameFromRefString
   :project: mirheo

.. doxygenfunction:: mirheo::readWholeFile
   :project: mirheo

.. doxygenfunction:: mirheo::writeToFile
   :project: mirheo

.. doxygenfunction:: mirheo::assertType
   :project: mirheo

.. doxygenstruct:: mirheo::TypeLoadSave
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::TypeLoadSaveNotImplemented
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ConfigArray
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ConfigObject
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ConfigValue
   :project: mirheo
   :members:
