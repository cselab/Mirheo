The main folder: mpi-dpd
=======

This is the "main" folder containing the object files orchestrating the various kernels.
The generated object files are the following ones:

* common.o: global simulation parameters, common datastructures like device arrays, cell lists etc.
* contact.o: computation of the contact/lubrication force across "touching" solute particles
* containers.o: particle arrays, collections encapsulating the data of RBCs and CTCs
* dpd.o: code coordinating the computation of cuda-dpd
* fsi.o: computation of the "flow-structure interaction" force between solvent and solute particles
* io.o: data dumps in XYZ, PLY (for cells), H5Part, HDF5 structured grids
* main.o: home sweet home
* minmax.o: computation of the extent of an array of RBCs or CTCs
* redistancing.o: computation of the distance transform for the implicit description of the wall geometry
* redistribute-particles.o: redistribution of the solvent across the MPI ranks once particles have moved
* redistribute-rbcs.o: redistribution of RBCs across MPI ranks once they have moved
* scan.o: computation of the prefix sum for the solvent cell lists count
* simulation.o: simulation "driver" coordinating the other object files, except for main.cu
* solute-exchange.o: exchange the "halo" solute particles across the MPI ranks close by to compute FSI and contact forces.
* solvent-exchange.o: exchange the "halo" solvent particles across the MPI ranks to compute the DPD interactions
* wall.o: computation of the particles interacting with the no-slip boundary conditions of the wall

Compiling uDeviceX
-------------
The makefile will check for a .cache.Makefile, that can be optionally put in this folder.
For example my .cache.Makefile on Piz Daint is

`h5part = 0
NVCC = nvcc -I$(CRAY_MPICH2_DIR)/include -L$(CRAY_MPICH2_DIR)/lib -I/scratch/daint/diegor/h5part/include -I$(HDF5_DIR)/include -I/users/diegor/vtk/install/include/vtk-6.2 -I/users/diegor/h5part/include/
CXX = CC $(CRAY_CUDATOOLKIT_POST_LINK_OPTS) $(CRAY_CUDATOOLKIT_INCLUDE_OPTS)   -L/users/diegor/h5part/lib -L$(HDF5_DIR)/lib -L/users/diegor/vtk/install/lib`

To clean uDeviceX entirely (mpi-dpd, cuda-dpd, cuda-rbc, cuda-ctc):
`make cleanall`

To just cleanup the mpi-dpd folder:
`make clean`

To compile:
`make -j`

Running uDeviceX
-----------
Running uDeviceX consists of these steps:

1. Generation of the geometry file (optional), the file should always be named `sdf.dat` (as Signed Distance Function).
See the folder `device-gen`.
2. Generation of the initial positioning of the RBCs and CTCs (optional). The IC files should be called
`rbcs-ic.txt and ctcs-ic.txt` See the folder `cell-placement`.
3. Execution of uDeviceX, for example `mpirun ./test 4 4 2 -walls -couette=1 -tend=5e4  -steps_per_dump=1000 -rbcs -contactforces`
4. Post processing of the simulation data (optional), for example `ls ./stress/* -rt1 | tail -n 50 | mpirun -n 32 -N 1 ../postprocessing/stress/stress -origin=0,0,5 -extent=192,192,85 -project=1,1,0 > stress-profile.txt`
