.. _dev-types:

Types
=====
Each channel in :any:`mirheo::DataManager` can have one of the types listed in the following xmacro:

.. doxygendefine:: MIRHEO_TYPE_TABLE__
   :project: mirheo


Host variants
^^^^^^^^^^^^^

.. doxygenstruct:: mirheo::DataTypeWrapper
   :project: mirheo
   :members:


The ``mirheo::TypeDescriptor`` variant contains a type of :any:`mirheo::DataTypeWrapper` that is in the type list.

..
   Not documented because of breathe warnings.
   .. doxygentypedef:: mirheo::TypeDescriptor
      :project: mirheo

Device variants
^^^^^^^^^^^^^^^

The ``mirheo::CudaVarPtr`` variant contains a pointer of a type that is in the type list.

..
   Not documented because of breathe warnings.
   .. doxygentypedef:: mirheo::CudaVarPtr
      :project: mirheo


Utils
^^^^^

.. doxygenfunction:: mirheo::typeDescriptorToString
   :project: mirheo

.. doxygenfunction:: mirheo::stringToTypeDescriptor
   :project: mirheo

..
   .. doxygenfunction:: mirheo::printToStr(int)
      :project: mirheo

