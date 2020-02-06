.. _user-testing:

.. role:: console(code)
   :language: console

Testing
#######

Mirheo can be tested with a set of regression tests (located in ``tests``) and unit tests (located in ``units``).

Regression tests
****************

Regression testing makes use of the `atest <https://gitlab.ethz.ch/mavt-cse/atest.git>`_ framework.
This can be installed as follows:

  .. code-block:: console

     $ git clone https://gitlab.ethz.ch/mavt-cse/atest.git
     $ cd atest
     $ make bin

  .. note::

     By default, this will install the atest executables in ``$HOME/bin`` folder.
     This location should be in your ``PATH`` variable

The regression tests are a set of python scripts.
They make use of additional dependencies:

- numpy
- trimesh
- mpi4py

Which can all be installed via ``pip``.
All tests can be run by typing:

  .. code-block:: console

     $ cd tests
     $ make test

  .. note::

     You need to install the tools before running the tests

Units tests
***********

Unit tests are compiled together with the `google-test framework <https://github.com/google/googletest>`_.
The unit tests are compiled by adding the option ``-DBUILD_TESTS=ON`` to cmake (see :ref:`user-install`).
The binaries are placed in the ``build`` folder.

  .. code-block:: console

     $ mir.make units
     $ cd build
     $ mir.make test

  .. note::

     You need to install the tools before running the unit tests


Double precision
****************

If compiled with ``DOUBLE_PRECISION=ON`` (see :ref:`user-install`), the reference files for the regression tests are inside the ``tests/test_data_double`` folder.
The tests can be run by typing:

  .. code-block:: console

     $ cd tests
     $ make test_double
