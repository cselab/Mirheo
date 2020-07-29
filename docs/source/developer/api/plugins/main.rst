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



Statistics and In-situ analysis Plugins
***************************************

These plugins do not modify the state of the simulation.
They are used to measure properties of the simulation that can be processed directly at runtime.
Their output is generally much lighter than dump plugins.
The prefered format is csv, to allow clean postprocessing from e.g. python.

.. doxygenclass:: mirheo::ObjStatsPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ObjStatsDumper
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::RdfPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::RdfDump
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::SimulationStats
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PostprocessStats
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::VacfPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::VacfDumper
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


.. doxygenclass:: mirheo::BerendsenThermostatPlugin
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


.. doxygenclass:: mirheo::OutletPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PlaneOutletPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::RegionOutletPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::DensityOutletPlugin
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::RateOutletPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ParticleChannelSaverPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::ParticleDragPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::PinObjectPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::PinRodExtremityPlugin
   :project: mirheo
   :members:


.. doxygenclass:: mirheo::TemperaturizePlugin
   :project: mirheo
   :members:



Debugging plugins
*****************

.. doxygenclass:: mirheo::ParticleCheckerPlugin
   :project: mirheo
   :members:
