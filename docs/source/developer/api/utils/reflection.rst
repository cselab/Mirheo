.. _dev-utils-reflection:

Reflection
==========

Type traits for their names and content.

Useful macros
-------------

.. code-block:: c++

    // Generates TypeName<Foo>. Remembers type's name.
    MIRHEO_TYPE_NAME(Foo, "Foo");

    // Shorter variant of MIRHEO_TYPE_NAME:
    MIRHEO_TYPE_NAME_AUTO(Foo);

    // Generates TypeName<Bar> and MemberVars<Bar>.
    // Remembers type's name and the order, names and types of its
    // member variables.
    MIRHEO_MEMBER_VARS(Bar, var1, var2, var3);


Details
-------

.. doxygenstruct:: mirheo::TypeName
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::MemberVars
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::MemberVarsAvailable
   :project: mirheo
   :members:
