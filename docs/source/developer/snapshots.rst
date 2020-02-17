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
- HDF5 files for particle and object vector data,
- .off files for meshes,
- other .dat files and similar.

The JSON files store the metadata of all particle vectors, interactions, integrators, plugins and other special objects like meshes or the ``Mirheo`` class instance.
The Mirheo setup and interdependency of objects can be represented as a directed acyclic graph (DAG).
To transform it into a tree-like JSON structure, we group objects by their categories (``Mesh``, ``ParticleVector``, ``Interaction``...), and use *reference strings* to link one object to another (e.g. ``"<TYPE with name=NAME>"``, ``using ConfigRefString = std::string;``).

The following is a preview of the two JSON objects (simulation and postprocess) from the interactions test case:

.. NOTE: If replacing this test case with another, don't forget to update comments in the corresponding .py files!

.. literalinclude:: ../../../tests/test_data/snapshot.ref.snapshot.interactions.txt
    :name: snapshot.ref.snapshot.interactions.txt
    :caption: `snapshot.ref.snapshot.interactions.txt`



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
Nevertheless, the 3rd party libraries we tried so far were not desirable for various reasons like: (a) huge git history, (b) complicated compilation procedures, (c) 32-bits numbers instead of 64-bit, (d) no differentiation between integers and floats, (e) incomplete implementations, (f) C++17 etc.


Snapshot saving and loading
---------------------------

Each C++ type that is potentially stored in a snapshot has to specialize the template class ``TypeLoadSave`` and implement its member functions:

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

This class is specialized for all important primitive types (``int``, ``float``...), simple structs (``float3``, ...), Mirheo-specific simple structs (interaction parameters), pointers and pointer-like types, template classes (``std::vector``, ``std::map``, ``mpark::variant``...), and for polymorphic classes like ``MirObject``, ``Mesh`` and their derived classes.
The only exception is the ``Mirheo`` class, which is responsible for initiating the saving or loading of a snapshot.
The template class pattern is used to increase type safety and to avoid any implicit conversions during saving or loading, i.e. avoiding that snapshotting logic of one type is used for some other type.

Referenceable types like ``MirObject``, ``Mesh`` and their derived classes are treated differently from simple types.
Their ``save`` and ``load`` functions must ensure that the objects are saved and loaded only once.
For developers, it is sufficient for the base class to inherit from ``AutoObjectSnapshotTag``.
A partial specialization of ``TypeLoadSave`` will take care of the rest.
The details can be found in the code.

Saving
^^^^^^

The ``MirObject`` and ``Mesh`` classes are also polymorphic types, so there are three things to keep in mind:

- there must be a virtual dispatch to the correct save function,
- there should be a mechanism to detect if an implementation of save functions is missing,
- there should be a way to extend base class's save function.

To address these three points, we organize our snapshot saving code in two functions:

- a protected non-virtual function ``ConfigObject _saveSnapshot(Saver& saver, [const std::string& category, ] const std::string& typeName);``
- a public virtual function ``void saveSnapshotAndRegister(Saver& saver);``

The former function saves any large data to disk and returns a ``ConfigObject`` that describes the object.
The latter function is a thin wrapper around ``_saveSnapshot`` which checks the dynamic type of the ``this`` pointer and registers the object.

Loading
^^^^^^^

To load a snapshot, the user passes the snapshot path to the ``Mirheo`` constructor.
The code in ``mirheo/core/snapshot.cpp`` parses the JSON files, creates all objects and invokes appropriate ``Mirheo::register*`` functions.
This way the loading phase is implemented as another front-end equivalent to the Python bindings.
Although the saving phase always requires the saver object to be passed, for brevity we enabled loader-free conversion of JSON values to simple and very common types.
See the ``parse`` function above and ``mirheo/core/utils/config.h`` for more details.

Loading objects from snapshots is more explicit than saving.
We must manually match type names to the class names and invoke the correct constructors.
To keep things clean and to avoid increasing the compilation time greatly, the code is organized into factory functions, one per category.


Attributes
----------

Sometimes it's useful to attach custom information to snapshots, such as the desired number of time steps.
The ``Mirheo`` object therefore implements a ``setAttribute(name, value)`` function that can add or update a user-defined attribute, and ``getAttribute*(name)`` functions to read the values.


Code organization
-----------------

The code is split into ``mirheo/core/utils/config.h`` which has no Mirheo-specific information, and ``mirheo/core/snapshot.h`` which contains information about ``MirObject`` classes and other.
To minimize the effect on compilation time, we use forward declarations, pass-by-reference and templates whenever possible.
The ``TypeLoadSave`` specialization for various parameter structs is done through a single partial specialization based on type reflection.
See ``mirheo/core/utils/reflection.h`` for more details.
