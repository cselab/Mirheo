.. _user-install:

.. role:: console(code)
   :language: console

Installation
############

uDeviceX
********

uDeviceX requires at least Kepler-generation NVIDIA GPU and depends on a few external tools and libraries:

- Unix-based OS
- NVIDIA CUDA toolkit version >= 8.0
- gcc compiler with c++11 support compatible with CUDA installation
- CMake version >= 3.8
- Python interpreter version >= 2.7
- MPI library
- HDF5 parallel library
- libbfd for pretty debug information in case of an error

With all the prerequisites installed, you can take the following steps to run uDeviceX:

#. Get the up-to-date version of the code:

   .. code-block:: console
      
      $ git clone --recursive https://github.com/cselab/uDeviceX.git udevicex
      
#. In most cases automatic installation will work correctly, you should try it in the first place.
   Navigate to the folder with the code and run the installation command:
   
   .. code-block:: console
      
      $ cd udevicex
      $ make install
    
   In case of any issues, check the prerequisites or try a more "manual" way:
    
   #. From the udevicex folder, create a build folder and run CMake:
   
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
            
         Note that in case you don't specify any capability, uDeviceX will be compiled for all supported architectures, which increases
         compilation time and slightly increases application startup. Performance, however, should not be affected.
      
   #. Now you can compile the code:
   
      .. code-block:: console
         
         $ make -j <number_of_jobs> 
      
      The library will be generated in the current build folder.
      
   #. A simple way to use uDeviceX after compilation is to install it with pip. Navigate to the root folder of uDeviceX
      and run the following command:
      
      .. code-block:: console
         
         $ pip install --user --upgrade .
         
         
#. Now you should be able to use the uDeviceX in your Python scripts:
      
   .. code-block:: python
        
      import udevicex
   


Tools
*****

Additional helper tools can be installed for convenience.
They are required for testing the code.

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
      In order to run on a cluster with a job scheduler (e.g. slurm), the ``--exec-cmd`` option should be set to the right command, such as ``srun``:
      
      .. code-block:: console
      
	 $ ./configure --exec-cmd <my-custom-command>

      The default value is ``mpiexec``


The tools will automatically load modules for installing, running and testing the code.
The modules and CMAKE flags can be customised by adding corresponding files in ``tools/config`` (see available examples).
The ``__default`` files can be modified accordingly to your system.

The installation can be tested by calling

   .. code-block:: console
        
      $ make test

The above command requires the  `atest <https://gitlab.ethz.ch/mavt-cse/atest.git>`_ framework (see :ref:`user-testing`).
