.. _dev-snapshots:

Snapshots
=========

Mirheo provides currently two different mechanism for storing the simulation state.
The `checkpointing` mechanism stores periodically the simulation state such as timestep index, particle data, object data, and other.
To restart a checkpoint, the simulation setup has to be known.
The so called `snapshotting` mechanism stores the whole simulation state, including the setup.
Snapshotting is a work-in-progress alternative to checkpointing that should or could at some point supersede it.

Snapshot format
---------------

The current implementation saves the whole simulation state as a single folder containing the following:
- two JSON files describing the simulation setup, one for simulation side, one for postprocessing side of Mirheo,
- HDF5 files (particle and object vector data),
- .off files (meshes),
- other .dat files and similar.

The JSON files store the metadata of all particle vectors, interactions, integrators, plugins and other special objects like meshes or the ``Mirheo`` class instance.
The Mirheo setup and interdependency of objects can be represented as a Directed Acyclic Graph.
To transform it into a tree-like JSON structure, we group objects by their categories (``Mesh``, ``ParticleVector``, ``Interaction``...), and use *reference strings* (``using RefString = std::string;``) to link one object to another.

The following is a preview of a simulation-side JSON object generated from an RBC simulation:

.. code-block:: javascript

   {
        "Mesh": [
            {
                "__type": "MembraneMesh",
                "name": "rbcMesh"
            },
            ...
        ],
       "ParticleVector": [
           {
               "__type": "MembraneVector",
               "name": "rbcOV",
               "mass": 1,
               "objSize": 1234,
               "mesh": "<MembraneMesh with name=rbcMesh>"
           },
           ...
       ],
       "Simulation": [
           {
               "__type": "Simulation",
               "name": "simulation",
               "particleVectors": [
                   "<MembraneVector with name=rbcOV>",
                   ...
               ],
               ...
           }
       ],
       "Mirheo": [
           {
               "__type": "Mirheo",
               "state": {
                   "__type": "MirState",
                   "domainGlobalStart": [0, 0, 0],
                   "domainGlobalSize": [4, 6, 8],
                   "dt": 0.10000000149011612,
                   "currentTime": 0,
                   "currentStep": 0
               },
               "simulation": "<Simulation with name=simulation>"
           }
       ]
   }


Internal representation of the JSON
-----------------------------------

Mirheo implements the JSON data structure in ``mirheo/core/utils/config.h`` as a ``ConfigValue`` class, a variant-like type that can store integers, floating point numbers, strings, arrays and dictionaries.
The following table shows the name of JSON data types and their aliases:

.. list-table:: Internal representation of JSON.
   :widths: 10 10 80
   :header-rows: 1

   * - Type
     - Alias
     - Note
   * - ``long long``
     - ``ConfigValue::Int``
     - We use single integer type for both unsigned and signed integers.
   * - ``double``
     - ``ConfigValue::Float``
     -
   * - ``std::string``
     - ``ConfigValue::String``
     -
   * - ``ConfigArray``
     - ``ConfigValue::Array``
     - Derives from ``std::vector<ConfigValue>`` and adds bound checks to ``operator[]``.
   * - ``ConfigObject``
     - ``ConfigValue::Object``
     - Derives from ``FlatOrderedDict<std::string, ConfigValue>``, a simple map implementation that preserves the insertion order. Adds bounds checks to ``operator[]``.
   * - ``ConfigValue``
     - N/A
     - A variant of the values above.

Originally we intended this not to be restricted to JSON, since it may be desirable to dump a Python code that recreates the simulation.
However, at this point, it seems like any JSON implementation would do.
Nevertheless, the 3rd party libraries we tried so far were not desirable for various reasons like: (a) huge git history, (b) complicated compilation procedures, (c) 32-bits numbers instead of 64-bit, (d) no differentiation between integers and floats, (e) not fully implemented etc.


Snapshot saving and loading
---------------------------

Each C++ type that is potentially stored in a snapshot has to specialize the template class ``TypeLoadSave`` and implements its member functions:

.. code-block:: C++

   template <typename T, typename Enable>
   struct TypeLoadSave
   {
       /// Store any data in files and prepare the ConfigValue describing the object.
       static ConfigValue save(Saver&, T& value);

       /// Context-free parsing. Only for simple types! (optional)
       static T parse(const ConfigValue&);

       /// Context-aware load.
       static T load(Loader&, const ConfigValue&);
   };

This class is specialized for all important primitive types (``int``, ``float``...), simple struct types (``float3``, ...), Mirheo data structures (interaction parameters), pointers and pointer-like types, template classes (``std::vector``, ``std::map``, ``mpark::variant``...), and for special types like ``MirObject`` and ``Mesh``.
The only exception is the ``Mirheo`` class, which is responsible for initiating the saving or loading of a snapshot.
The template class pattern is used to increase type safety and avoid any implicit conversions during saving or loading, i.e. avoiding that a save or load function of one type is used for another type.

Complex types like ``MirObject``, ``Mesh`` and their derived classes are treated differently than simple types.
As explained above, they are categorized, and may be referred to through ``RefString`` references.
Their ``save`` and ``load`` functions must ensure, with the help of ``Saver`` and ``Loader`` helper objects, that the objects are saved and loaded only once.
The details can be found in the code.

In practice, what is important is that ``MirObject`` and ``Mesh`` classes are polymorphic types.
For saving, each derived class must implement a public virtual function ``saveSnapshotAndRegister``, which saves any data on the disk, creates the ``ConfigObject`` description of the object, and registers itself in the saver.
To ensure that every class overrides this function, they check for the dynamic type of the ``this`` pointer.
However, that prevents us from reusing base class saving procedure in its derived classes.
For that reason, the ``saveSnapshotAndRegister`` function is implemented as a thin wrapper around a reusable protected function ``_saveSnapshot``, which only saves the data and creates the ``ConfigObject``, but performs no type check or registration.
This ``_saveSnapshot`` function can then be used by the derived classes to simplify the code and reduce potential bugs.

To load a snapshot, the user passes the snapshot path to the ``Mirheo`` constructor.
The code in ``mirheo/core/snapshot.cpp`` then parses the JSON files, creates all objects and invokes appropriate ``Mirheo::register*`` functions.
This way the loading phase is implemented as a front-end equivalent to Python bindings.
Although the saving phase always requires the saver object to be passed, for brevity we enabled loader-free loading of simple types from JSON objects.
See the ``parse`` function above.

Implementation details
----------------------

The code is split into ``mirheo/core/utils/config.h`` which has no Mirheo-specific information, and ``mirheo/core/snapshot.h`` which contains information about ``MirObject`` classes and other.
To minimize the effect on compilation time, we use forward declarations, pass-by-reference and templates if possible.
The ``TypeLoadSave`` specialization for various parameter structs is done through a singel partial specialization based on type reflection. See ``mirheo/core/utils/reflection.h`` for more details.
