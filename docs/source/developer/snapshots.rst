.. _dev-snapshots:

Snapshots
=========

Mirheo provides currently two different mechanism for storing the simulation state.
The `checkpointing` mechanism stores periodically the simulation state such as timestep index, particle data, object data, and other.
To restart a checkpoint, the simulation setup has to be known.
The so called `snapshotting` mechanism stores the whole simulation state, including the setup.
Snapshotting is a work-in-progress alternative and superset of checkpointing.

Snapshot format
---------------

The current implementation saves the whole simulation state as a single folder containing the following:

- two JSON files describing the simulation setup, one for simulation side, one for postprocessing side of Mirheo,
- HDF5 files for particle and object vector data,
- .off files for meshes,
- other .dat files and similar.

The JSON files store the metadata of all particle vectors, interactions, integrators, plugins and other special objects like meshes or the ``Mirheo`` class instance.
The Mirheo setup and interdependency of objects can be represented as a graph.
To transform it into a tree-like JSON structure, we group objects by their categories (``Mesh``, ``ParticleVector``, ``Interaction``...), and use *reference strings* to link one object to another (e.g. ``ConfigRefString ref = "<TYPE with name=NAME>"``, where ``ConfigRefString`` is an alias of ``std::string``).

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

This class is already specialized for all important primitive types (``int``, ``float``...), simple structs (``float3``, ...), Mirheo-specific structs (interaction parameters), pointers and pointer-like types, template classes (``std::vector``, ``std::map``, ``mpark::variant``...), and for polymorphic classes like ``MirObject``, ``Mesh`` and their derived classes.
The only exception is the ``Mirheo`` class, which initiates the saving or loading of all other objects.
The template class pattern is used to increase type safety and to avoid any implicit conversions during saving or loading, i.e. to avoid that snapshotting logic of one type is used for some other type.

The ``Saver`` and ``Loader`` objects listed in the API above, among other things, provide helper APIs for serialization and unserialization of objects, and access to saving and loading contexts. The contexts store any information useful for saving and loading, such as the snapshot path and a Mirheo MPI communicator.

Saving
^^^^^^

String-referenceable types like ``MirObject``, ``Mesh`` and their derived classes are treated differently from simple types.
Their ``TypeLoadSave<>::save`` function must ensure that the objects are saved only once.
This is achieved by registering them in the saver, and instead of returning the ``ConfigObject`` itself, a ``ConfigRefString`` is returned.
For developers, it is sufficient to make these classes inherit from ``AutoObjectSnapshotTag``.
A partial specialization of ``TypeLoadSave`` will take care of the rest.
The details can be found in the code.

The ``MirObject`` and ``Mesh`` classes are also polymorphic types, so there are three things to keep in mind:

- there must be a virtual dispatch to the correct save function,
- there should be a mechanism to detect if an implementation of save functions is missing,
- there should be a way to reuse base class's save function in the derived classes.

To address these three points, we organize our snapshot saving code of a polymorphic type ``T`` in two member functions:

- a protected non-virtual function ``ConfigObject T::_saveSnapshot(Saver& saver, [const std::string& category, ] const std::string& typeName);``
- a public virtual function ``void T::saveSnapshotAndRegister(Saver& saver);``

The former function saves any large data to disk and returns a ``ConfigObject`` that describes the object.
The latter function is a single-line wrapper around ``T::_saveSnapshot`` that registers the object in the saver and checks if the dynamic type of the ``this`` pointer is exactly ``T``.
For a class ``U`` that derives from ``T``, the function ``U::_saveSnapshot`` may and is encouraged to use ``T::_saveSnapshot``.
On the other hand, the function ``T::saveSnapshotAndRegister`` is the mechanism of detecting unimplemented save functions, and cannot be used from derived classes.
See the code for details.


Loading
^^^^^^^

To load a snapshot, the user passes the snapshot path to the ``Mirheo`` constructor, which is then passed to ``loadSnapshot`` in ``mirheo/core/snapshot.cpp``.
The load function creates all objects and invokes appropriate ``Mirheo::register*`` functions.
This way the loading phase effectively reproduces the front-end steps done through Python bindings.
Although the saving phase always requires the saver object to be passed, for brevity we enabled loader-free conversion of JSON values to simple and very common types such as integers, strings and floats.
Search for the ``parse`` function in ``mirheo/core/utils/config.h`` for more details.

Loading objects from snapshots is more explicit than saving.
We must manually match type names to the classes and invoke the correct constructors.
To keep things clean and to avoid increasing the compilation time greatly, the code is organized into factory functions, one per every category (apart from trivial categories ``Simulation``, ``Postprocess`` and ``Mirheo``).


Attributes
----------

It is sometimes useful to attach custom information to snapshots, e.g. the desired number of time steps.
The ``Mirheo`` object therefore provides a ``setAttribute(name, value)`` function that adds or updates a user-defined attribute, and ``getAttribute*(name)`` functions to read the attribute values.


Code organization
-----------------

The code is split into ``mirheo/core/utils/config.h`` which has no Mirheo-specific information, and ``mirheo/core/snapshot.h`` which contains information about ``MirObject`` classes and other.
To minimize the effect on compilation time, we use forward declarations, pass-by-reference and move as much code as possible to .cpp files.
The ``TypeLoadSave`` specializations for various parameter structs are done with the help of manually-implemented reflection.
See ``mirheo/core/utils/reflection.h`` for more details.
