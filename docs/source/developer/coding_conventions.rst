Coding Conventions
==================

In this section we list a guidelines to edit/add code to Mirheo.

Naming
------

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

Class names (and all types) have a camelCase format and start with an upper case letter:

.. code-block:: c++

   class MyClass;         // OK
   using MyIntType = int; // OK
   class My_Class;        // Not OK

Functions and public member functions follow the same rules as local variables.
They should state an action and must be meaningfull, especially when they are exposed to the rest of the library.

.. code-block:: c++

   Mesh readOffFile(std::string fileName); // OK
   Mesh ReadOffFile(std::string fileName); // not OK
   Mesh read(std::string fileName);        // not precise enough naming out of context
