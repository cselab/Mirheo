# YMeRo

[![Documentation Status](https://readthedocs.org/projects/ymero/badge/?version=latest)](https://ymero.readthedocs.io/en/latest/?badge=latest)

Computational Microfluidics

YMeRo is a GPU high-performance and high-throughput code aimed at simulation of flows at milli- and microscales.
The code uses Dissipative Particle Dynamics method to describe the liquid and its interaction with cells and other bodies.

For more information, please refer to the online documentation: http://ymero.readthedocs.io/


## Changelog

### unreleased 

* support for VOLTA architecture

### v0.9.6

* bug fix: LJ potential had swapped epsilon and sigma

### v0.9.5

* separate sdf grid implementation into more general core/field
* field can be initialized from std::function
* pressure plugin uses region

### v0.9.4

* fix: stress free state can be used when the cell is grown
* fix: MembraneMesh wrapper needs GPU

### v0.9.3

* fix: the stress entries are now cleared before forces; could be cleared more by other interaction handlers
* use ymero state inside the simulation objects; do not have current time, step and dt separate in simulation

### v0.9.2

* add filtered initial conditions: allows custom regions in space to initialise uniform density particles

### v0.9.1

* add sphere initial condition
* add plugin to compute total virial pressure from stresses per particle for a given pv

### v0.9.0

Add common YmrState object.
This changes the interface only slightly due to the python wrapping:
* the time step is now passed to the coordinator constructor
* the time step is not passed to any other object

### v0.8.0

* add checkpoint for permanent channels
* extra data managers are aware of the type

### v0.7.1

* add permanent channels in extra data manager

### v0.7.0

* reorganise memebrane interaction kernels:
  * bending force kernels now separated from other
  * 2 parameter strctures
* add Juelicher bending model 
* add force saver plugin to save forces in channels

### v0.6.1

* rename uDeviceX to YMeRo
* synchronzation bug fix

### v0.6.0
 
* add plugin for magnetic orientation of rigid bodies

### v0.5.1

* make the stress channel name customizable

### v0.5.0

* add stress computation + tests
* perf improvement in sdf
* minor perf improvement in pairwise kernels

### v0.4.2

* compile some units
* use gtest

### v0.4.1

* change interface for wall oscillation: period is in dpd units now

### v0.4.0

* add hdf5 support for mesh dump
* allow for extra channels to be dumped together with the mesh

### v0.3.1

* add compile time switch for CUDA>9 support
* add extra force plugin

### v0.3.0

* proper MPI init and finalize
* communicator can be passed from python

### v0.2.1

* add tools python submodule

### v0.2.0

* add checkpoint/restart support for object vectors
* dumped in xdmf+hdf5 format: restart files can be viewed

### v0.1.4

* add bounce tests
* wip: xdmf support for restart

### v0.1.3

* bug fix: bounce back with substep integration
* xdmf reader

### v0.1.2

* dump walls in xdmf format
* add tests for bounce back on mesh + rigid ellipsoids

### v0.1.1

* sdf fix: merged sdf before splittinf frozen particles

### v0.1.0

* set up versioning
