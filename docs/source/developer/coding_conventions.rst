Coding Conventions
==================

In this section we list a guidelines to edit/add code to Mirheo.

Naming
------

Variables
^^^^^^^^^
Local variable names and paramters follow camelCase format starting with a lower case:

.. code-block:: c++

   int myInt;   // OK
   int MyInt;   // not OK
   int my_int;  // not OK


Member variable names inside a ``class`` (not for ``struct``) have a trailing ``_``:

.. code-block:: c++

   class MyClass
   {
   private:
	int myInt_;   // OK
	int myInt;    // not OK
	int my_int_;  // not OK
   };

Types, classes
^^^^^^^^^^^^^^

Class names (and all types) have a camelCase format and start with an upper case letter:

.. code-block:: c++

   class MyClass;         // OK
   using MyIntType = int; // OK
   class My_Class;        // Not OK


Functions
^^^^^^^^^

Functions and public member functions follow the same rules as local variables.
They should state an action and must be meaningfull, especially when they are exposed to the rest of the library.

.. code-block:: c++

   Mesh readOffFile(std::string fileName); // OK
   Mesh ReadOffFile(std::string fileName); // not OK
   Mesh read(std::string fileName);        // not precise enough naming out of context

private member functions have an additional `_` in front:

.. code-block:: c++

   class MyClass
   {
   public:
       void doSomething();
   private:
       void _doSubTask();   // OK
       void doSubTask();    // Not OK
       void _do_sub_task(); // Not OK
   };

namespaces
^^^^^^^^^^

Namespaces are written in ``snake_case``.
This allows to distinguish them from class names easily.


Coding practices
----------------

Use modern C++ whenever possible.
Refer to `c++ core guidelines <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>`_ up to C++ 14.
Some exceptions in Mirheo:

- Do not fail with exceptions. Mirheo crashes with the ``die`` method from the logger. This will print the full stacktrace.
  

Style
-----

The indentation uses 4 spaces (no tabs).
Here are a few coding examples of the style:

.. code-block:: c++

   // loops
   for (int i = 0; i < n; ++i)
   {
       // multi line commands
       ...
   }
   
   for (int i = 0; i < n; ++i)
       // one line command

   // if
   if (condition)
       doThat();
   else
       doThis();

   // for multi line, all entries must have braces
   if (condition)
   {
       doThat();
       andThis();
   }
   else
   {
       doThis();
   }
   
   
More can be found directly in the code.

Includes
--------

Every header file must include all files required such that it compiles on its own.
The includes must be grouped into 3 groups with the following order:

1. local files (relative path)
2. mirheo files (path relative to mirheo src dir)
3. external libraries and std library

Each subgroup must be sorted alphabetically.
The first group has the quotes style while the other groups must use bracket style.

Example:

.. code-block:: c++

   #include "data_manager.h"

   #include <mirheo/core/containers.h>
   #include <mirheo/core/datatypes.h>
   #include <mirheo/core/mirheo_object.h>
   #include <mirheo/core/utils/pytypes.h>

   #include <memory>
   #include <string>
   #include <vector>
