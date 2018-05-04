.. _user-install:

Installation
############

uDeviceX requires at least Kepler-generation NVIDIA GPU and depends only on a few external tools and libraries:

- Unix-based OS
- NVIDIA CUDA toolkit version >= 8.0
- gcc compiler with c++11 support compatible with CUDA installation
- MPI library
- HDF5 parallel library
- libbfd for pretty debug information in case of crash

With all the prerequisites installed, you can take the following steps to run uDeviceX:

#. Get the up-to-date version of the code:

   .. code-block:: console
      
      $ git clone https://github.com/dimaleks/uDeviceX.git udevicex

#. Navigate to the makefile folder and create a makefile for your machine:

   .. code-block:: console
      
      $ cd udevicex/makefiles
      $ cp Makefile.settings.linux Makefile.settings.$(hostname)
      $ your-favourite-editor Makefile.settings.$(hostname)
   
   The default settings will not probably work for you, so you will have to modify the makefile.
   Follow the instructions in the file and change your settings accordingly.
   
#. Now go the apps folder and compile the two executables, udevicex and genwall (used to prepare the geometry for the simulations):

   .. code-block:: console
      
      $ cd ../apps
      $ make -j -C ./genwall
      $ make -j -C ./udevicex 
   
   The executables will be generated in apps/udevicex/udevicex and apps/genwall/genwall
