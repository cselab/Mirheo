.. _user-install:

.. role:: console(code)
   :language: console

Installation
############

Mirheo
******

Mirheo requires at least Kepler-generation NVIDIA GPU and depends on a few external tools and libraries:

- Unix-based OS
- NVIDIA CUDA toolkit version >= 9.2
- gcc compiler with c++14 support compatible with CUDA installation
- CMake version >= 3.8
- Python interpreter version >= 3.4
- MPI library
- HDF5 parallel library
- libbfd for pretty debug information in case of an error

.. note::

   The code has been tested with  ``mpich-3.2.1``, ``mpich-3.3.1`` and ``openmpi-3.1.3``.
   A known bug in ``mpich-3.3`` causes Mirheo to deadlock, use another version instead.
  
With all the prerequisites installed, you can take the following steps to run Mirheo:

#. Get the up-to-date version of the code:

   .. code-block:: console
      
      $ git clone --recursive https://github.com/cselab/Mirheo.git mirheo
      
#. In most cases automatic installation will work correctly, you should try it in the first place.
   Navigate to the folder with the code and run the installation command:
   
   .. code-block:: console
      
      $ cd mirheo
      $ make install
    
   In case of any issues, check the prerequisites or try a more "manual" way:
    
   #. From the mirheo folder, create a build folder and run CMake:
   
      .. code-block:: console
         
         $ mkdir -p build/
         $ cd build
         $ cmake ../
      
      If CMake reports some packages are not found, make sure you have all the prerequisites installed and corresponding modules loaded.
      If that doesn't help, or you have some packages installed in non-default locations,
      you will need to manually point CMake to the correct locations.
      
      See CMake documentation for more details on how to provide package installation files.
      
      .. note::
         On CRAY systems you may need to tell CMake to dynamically link the libraries by the following flag:
         
         .. code-block:: console
         
            $ cmake -DCMAKE_EXE_LINKER_FLAGS="-dynamic" ../
            
      .. note::
         Usually CMake will correctly determine compute capability of your GPU. However, if compiling on a machine without a GPU
         (for example on a login node of a cluster), you may manually specify the compute capability (use your version instead of 6.0):
         
         .. code-block:: console
         
            $ cmake -DCUDA_ARCH_NAME=6.0 ../
            
         Note that in case you don't specify any capability, Mirheo will be compiled for all supported architectures, which increases
         compilation time and slightly increases application startup. Performance, however, should not be affected.
      
   #. Now you can compile the code:
   
      .. code-block:: console
         
         $ make -j <number_of_jobs> 
      
      The library will be generated in the current build folder.
      
   #. A simple way to use Mirheo after compilation is to install it with pip. Navigate to the root folder of Mirheo
      and run the following command:
      
      .. code-block:: console
         
         $ pip install --user --upgrade .
         
         
#. Now you should be able to use the Mirheo in your Python scripts:
      
   .. code-block:: python
        
      import mirheo
   

Compile Options
***************

Additional compile options are provided through ``cmake``:

* ``MEMBRANE_DOUBLE:BOOL=OFF``: Computes membrane forces (see :any:`MembraneForces`) in double prcision if set to ``ON``; default: single precision
* ``ROD_DOUBLE:BOOL=OFF``:  Computes rod forces (see :any:`RodForces`) in double prcision if set to ``ON``; default: single precision
* ``USE_NVTX:BOOL=OFF``: Add NVIDIA Tools Extension (NVTX) trace support for more profiling informations if set to ``ON``; default: no NVTX


Tools
*****

Additional helper tools can be installed for convenience and are required for testing the code.

Configuration
-------------

The tools will automatically load modules for installing and running the code.
Furthermore, CMake options can be saved in those wrapper tools for convenience.
The list of modules and cmake flags can be customised by adding corresponding files in ``tools/config`` (see available examples).
The ``__default`` files can be modified accordingly to your system.

Installation
------------

The tools can be installed by typing:

   .. code-block:: console
        
      $ cd tools/
      $ ./configure
      $ make install

 
   .. note::
      By default, the tools are installed in your ``$HOME/bin`` directory.
      It is possible to choose another location by setting the ``--bin-prefix`` option:
      
      .. code-block:: console
      
	 $ ./configure --bin-prefix <my-custom-tools-location>


   .. note::
      In order to run on a cluster with a job scheduler (e.g. slurm), the ``--exec-cmd`` option should be set to the right command (e.g. ``srun``):
      
      .. code-block:: console
      
	 $ ./configure --exec-cmd <my-custom-command>

      The default value is ``mpiexec``


After installation, it is advised to test the tools by invoking

   .. code-block:: console
        
      $ make test

The above command requires the `atest <https://gitlab.ethz.ch/mavt-cse/atest.git>`_ framework (see :ref:`user-testing`).


Tools description
-----------------

mir.load
~~~~~~~~

This tool is not executable but need to be sourced instead.
This simply contains the list of of possible modules required by Mirheo.
``mir.load.post`` is similar and contains modules required only for postprocessing as it migh conflict with ``mir.load``. 


mir.make
~~~~~~~~

Wrapper used to compile Mirheo.
It calls the ``make`` command and additionally loads the correct modules and pass optional CMake flags.
The arguments are the same as the ``make`` command.

mir.run
~~~~~~~

Wrapper used to run Mirheo.
It runs a given command after loading the correct modules.
Internally calls the ``--exec-cmd`` passed during the configuation.
Additionally, the user can execute profiling or debugging tools (see ``mir.run --help`` for more information).
The parameters for the exec-cmd can be passed through the ``--runargs`` option, e.g.

    .. code-block:: console

       $ mir.run --runargs "-n 2" echo "Hello!"
       Hello!
       Hello!

Alternatively, these arguments can be passed through the environment variable ``MIR_RUNARGS``:

    .. code-block:: console

       $ MIR_RUNARGS="-n 2" mir.run echo "Hello!"
       Hello!
       Hello!

The latter use is very useful when passing a common run option to all tests for example.


mir.post
~~~~~~~~

Wrapper used to run postprocess tools.
This is different from ``mir.run`` as it does not execute in parallel and can load a different set of modules (see ``mir.load.post``)

mir.avgh5
~~~~~~~~~

a simple postprocessing tool used in many tests.
It allows to average a grid field contained in one or multiple h5 files along given directions.
See more detailed documentation in 

    .. code-block:: console

       $ mir.avgh5 --help

mir.restart.id
~~~~~~~~~~~~~~

Convenience tool to manipulate the restart ID from multiple restart files.
See more detailed documentation in 

    .. code-block:: console

       $ mir.restart.id --help


