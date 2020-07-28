.. _dev-plugins-main:

List of plugins
===============

Dump Plugins
************

These plugins do not modify the state of the simulation.
They can be used to dump selected parts of the state of the simulation to the disk.

.. doxygenclass:: mirheo::Average3D
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::AverageRelative3D
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::UniformCartesianDumper
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::MeshPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::MeshDumper
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ParticleSenderPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleDumperPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ParticleWithMeshSenderPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleWithMeshDumperPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::XYZPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::XYZDumper
   :project: mirheo
   :members:



Statistics Plugins
******************

These plugins Do not modify the state of the simulation.
They are used to measure properties of the simulation that can be processed directly at runtime.
Their output is generally much lighter than dump plugins.
The prefered format is csv, to allow clean postprocessing from e.g. python.

.. doxygenclass:: mirheo::ObjStatsPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjStatsDumper
   :project: mirheo
   :members:




Modifier plugins
****************

These plugins add more functionalities to the simulation.

.. doxygenclass:: mirheo::AddForcePlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::AddTorquePlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::AnchorParticlesPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::AnchorParticlesStatsPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::DensityControlPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PostprocessDensityControl
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ParticleDisplacementPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ExchangePVSFluxPlanePlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ForceSaverPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ImposeProfilePlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ImposeVelocityPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::MagneticOrientationPlugin
   :project: mirheo
   :members:
