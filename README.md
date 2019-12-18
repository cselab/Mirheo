# Mirheo

[![Documentation Status](https://readthedocs.org/projects/mirheo/badge/?version=latest)](https://mirheo.readthedocs.io/en/latest/?badge=latest)

Computational Microfluidics

Mirheo is a GPU high-performance and high-throughput code aimed at simulation of flows at milli- and microscales.
The code uses Dissipative Particle Dynamics method to describe the liquid and its interaction with cells and other bodies.

For more information, please refer to the online documentation: http://mirheo.readthedocs.io/

## Installation

```sh
git clone --recursive https://github.com/cselab/Mirheo
cd Mirheo
make install
```

## Changelog

### unreleased

* Do not create checkpoint folder when not needed
* **bug fix** 
* **internal changes**
  * state is private
  * add -Wconversion for `.cpp` files (not `.cu` for now because of thrust)

### v1.3.2

* **bug fix**: pass substep with random fluctuations had bias rng
* prepend macros with `MIRHEO_`
* **interface changes**:
  * average flow plugins: does not need type
  * changed channel names to a more consistent version
  * add list of reserved channel names into docs
* **internal changes**:
  * more modern CMakeLists
  * remove unneeded includes
  * reduce compilation time: do not include `std::chrono`
  * reduce compilation time: do not bind all plugin classes

### v1.3.1

* **interface changes**
  * substep integrator now takes a list of interactions 
  * ObjStats plugin dumps additional type id field if activated
  * dump particles plugin does not need type infos
  * dump particles plugin supports object and rod data
  * remove dump particles rod plugin as it would now be redundant
* add membrane with type id IC
* add filtered membrane forces, which can be used in coordination with the type ids field
* make Mirheo more usable as a library:
  * changed the directory structure
  * added namespace `mirheo`

### v1.3.0

* **Add double precision support**
  can be compiled with `CMAKE_FLAGS="-DDOUBLE_PRECISION=ON"  mir.make`

* **bug fixes**:
  * fixed cell lists sizes with domain sizes not multiple of rc
  * fix in reverse force exchanger
  * objectBelonging: split all persistent channels instead of positions and velocities only
  
* add `PlaneOutletPlugin` for removing particles that cross a given plane
* add `ParticlePortal` plugin that transfers standalone particles from one Mirheo instance to another
* add `ObjectDeleter` helper class for removing marked objects

Work in progress:
* adding `ObjectPortal` plugin that transfers objects from one Mirheo instance to another
* adding `ObjectsToParticles` plugin that destroys objects and transforms them into standalone particles once a condition is met

### v1.2.1

* support for different bounce kinds: every bouncer can be used with:
  * bounce back
  * bounce maxwell

* **Interface changes**:
  * need bounce kernel parameter for every bouncer

### v1.2.0

* **Interface changes**:
  * add function to log compile time options; this can be useful to make sure what was run exactly
  * generic Pairwise forces; removed all old pairwise classes from the interface
  * use `kBT` everywhere instead of `kbt`
  * use `number_density` instead of `density`
  * rename `mirheo` class coordinator to `Mirheo`

* **internal changes** 
  * refactored pairwise interaction drivers
  * more consistent naming


### v1.1.2

* **internal changes**: 
  * enable and fix warnings
  * fix initialization bugs
  * use forked [cselab/cuda_variant](https://github.com/cselab/cuda_variant)
  * refactor interaction manager: one for each interaction stage

### v1.1.1

* **interface changes**: new field `requireShift` in restart files. Newer versions will not be able to read older restart files.
* add unit tests for restart
* **internal changes**:
  * refactored restart for particle vectors
  * fix: test for ROV
  * fix: correctly assigns and shift data during restart IC

### v1.1.0

* **interface changes**:
  * no checkpointEvery in registerParticleVector or belongingChercker
* **internal changes**:
  * refactoring of packers
  * add persistent channels support for rod bisegment data
  * allow silent log
  * pack arguments: logInfo and checkpointInfo for better readability
* add unit tests for packers:
  * redistribute (particles, objects, rods)
  * exchange (particles, objects)
* test for bounce back reverse exchange:
  * rigid
  * membrane mesh
  * rigid mesh

### v1.0.0

* **rename to Mirheo** 
* add spin based state update for rods with polymorphic states (not validated yet)
* **fix** was a possible deadlock for many nodes / many objects suspensions; see commit 4b65e00eba8cf163db3e99e1730e6f16c12fcd26
* add Capsule Rigid Vectors together with bonce back and belonging checkers
* tests: renaming, semantic

### v0.17.0

* add CylinderRigidObjectVectors (CROVs)
* add bounce back on rods and CROVs
* add belonging checker for rods and CROVs
* **tests** belonging checkers
* **internal changes**:
  * do not rely on hash for mpi tags in plugins
  * generic kernels in belonging and bounce for RigidObjectVectors with analytical shapes 

### v0.16.3

* **fix** MPI cleanup
* **fix** restart in pvs now supports extra data
* **fix** checkpoint in pvs also shifts the extra data
* **fix** dump plugins for xyz, ply and grid h5 timestamp based on simulation time (hence continues after restart)
* **fix** float4 data is correctly dumped in checkpoint
* **minor fix** stats plugin works for 0 particles
* **interface changes** anchor particle plugin accepts multiple anchors
* add pin rod extremity plugin

### v0.16.2

* **fix** cleanup streams
* **fix** more robust filename creation
* **fix** more robust parameter passing to rod forces factory
* **interace chanes** rod forces:
  * change `omega0` to `kappa0`
  * remove `k_visc`
  * split `k_bounds` into `k_s_center` and `k_s_frame`
  * can optionally save energies and polymorphic states
* extend rod IC
* implement polymorphic states

### v0.16.1

* add interaction rod-rigid
* **fix** free communicator in plugins

### v0.16.0

* add `particleChecker` plugin
* **interface changes**: average flow plugin does not support float8 entries anymore due to internal changes; use float4 instead
* **internal changes**
  * particles are now separated into 2 arrays: positions and velocities; this will simplify packing/unpacking kernels
  * positions and velocities are stored as other quantities in dataManagers
  * old particles are only positions
  * vertices are only positions

### v0.15.0

* **output change** 
  * restart time-stamps are now continuing from the previous simulation
  * output dump files start at t=0
* **fix** in TextIO::reader
* internal changes:
  * currentStep is long long -> allow longer simulations
  * forces are stored as generic arrays -> simplifies a bit cell lists

### v0.14.4

* **interface change** pass rod width to rod ic
* add bound viscous forces to rod

### v0.14.3

* **fix** issues #38 and #54
* safe check: cannot register or set objects to coordinator after setup

### v0.14.2

* **interface changes** Integrator: SubStepMembrane -> SubStep
* SubStep integrator supports rods
* add anchor_particle plugin
* internal changes:
  * ids are stored in 64 bits integers

### v0.14.1

* Add "rod aware" LJ repulsion interactions
* **interface change** DPD, MDPD and LJ interactions have only one constructor each
* Safer parameters in interaction factory: treat unread parameters

### v0.14.0

* Add elastic rod interactions: bounds, bending and torsion
* Add corresponding unit test
* Add corresponding regression tests
* Add particle drap plugin
* **fix** (minor) dependencies for tools installation

### v0.13.0

* **compilation changes** require `c++ 14` and `cuda 9.2`
* **additional extern code** include `src/extern/mpark`
* **interface change** linear EOS in SDPD needs an extra parameter `rho_0`
* **fix** SDPD: correct use of mass density
* checkpoint and restart interaction random state
* internal changes:
  * cleanup some implementations from interfaces
  * adding rod vectors
  * remove xmacros for channel types
  * use mpark::variant for membrane interaction

### v0.12.3

* **interface changes**
  * use checkpoint instead of restart in Ymero coordinator constructor
  * add checkpoint_mode parameter to Ymero coordinator constructor
* checkpoint/restart mechanism is more uniform

### v0.12.2

* densities computed from pairwise interactions are now mass density
* internal changes: 
  * add fetcher with mass
  * add type trait for self interacting pairwise kernels

### v0.12.1

* add support of SDPD/MDPD for objects
* **fix** belonging checkers for halo objects: needed rigid motions
* **fix** reverse exchanger data taken from halo for the extra data

### v0.12.0

* **fix** `mirheo.restart()` will now indeed restart the simulation
* **interface change** Renamed and moved parameter for MPI communicator in the mirheo constructor
* **interface change** Density interaction needs a density kernel
* add SDPD interactions
* add more density kernels
* **docs**: add tutorials
* **docs**: minor fixes
* **internal**:
  * tests are more robust to DPD noise
  * proper rng per time step in DPD/MDPD 
* add density outlet plugin
* add rate outlet plugin
* add experimental version of density control plugin

### v0.11.5

* support for RDMA communication
* membrane fluctuation forces: use uniform distribution instead of gaussian so that it is bounded
* membrane forces: more robust computation of Lim forces
* internal changes: 
  * simplify interface of interaction manager
  * pairwise forces can contain only host memory variable and pass a separate handle to gpu
  * pairwise interaction: compiled separately when template instances are created

### v0.11.4

* **fix** Lim shear forces bug: triangle invariant
* **fix** reset accumulated quantities in Juelicher ADE model
* **fix** check MPI send requests

### v0.11.3

* **fix** Juelicher bending  forces sign
* **fix** Lim shear forces sign
* add possibility to compute membrane forces in double perecision

### v0.11.2

* **fix** wall thickness is larger when using MDPD
* internal changes:
  * cell lists do not know about special channels
  * add interaction manager to manage interaction channels

### v0.11.1

* **fix** bounce back on multiple nodes for objects
* minor fixes in documentation

### v0.11.0

* add Lim model for shear forces on membranes
* add corresponding tests

### v0.10.7

* **interface change** membrane interactions take only one constructor (see docs)
* **fix** warp reductions: all threads participate
* internal changes:
  * cosmetic changes: more consistent namespace style
  * membrane interaction takes a functor for dihedral computations
  * refactor membrane interactions: take functors for shear and bending forces

### v0.10.6

* Internal change: add before cell list plugin task
* **fix**: pvs exchanger plugin: new particles are now correctly redistributed
* **fix**: velocity inlet plugin: new particles are now correctly redistributed
* **fix**: communication of persistent data to empty pvs
* **fix**: average plugin asynchronous message to postprocessing

### v0.10.5

* add displacement plugin
* add radial velocity control plugin 
* **fix** reordering of extra data in celllists
* **fix** pvsExchanger plugin also copies persistent channels
* add test for pvsExchanger
* add test for displacement plugin
* internal changes:
  * packers can copy to another packer
  * defaultStream in `core/utils/cuda_common.h`

### v0.10.4

* add velocity inlet plugin
* add very simple CPU marching cubes implementation in core
* clean up units

### v0.10.3

* add wall force collector plugin
* automated support of multiple GPUs on single nodes
* **fix** in bounce
* remove deprecate warings for python 3.7

### v0.10.2

* Adds support for different stress free shape than original mesh

### v0.10.1

* add plugin to save a channel of extra particle data (useful for intermediate quantities such as densities in MDPD)
* **fix** reordering of persistent extra channels in primary cell lists
* **fix** use local cell lists instead of primary ones in halo exchanger

### v0.10.0

* Add _MDPD_ interaction (**walls and solvent fully supported only**)
* internal changes:
  * generic pairwise interaction fetching
  * 2-steps interaction support: extended task dependency graph
  * cell lists are aware of which channels to clear, accumulate and reorder
  * wip: more general object reverse exchangers
* **interface change**: make frozen walls takes a list of interactions
* **interface change**: make frozen rigid takes a list of interactions

### v0.9.7

* support for VOLTA architecture
* internal changes
  * generic pairwise interaction output: accumulators
  * generic pairwise_interaction: pass views
  * cell lists produce views; cellinfos don not know about particles and forces
  * less magic numbers

### v0.9.6

* **fix**: LJ potential had swapped epsilon and sigma

### v0.9.5

* separate sdf grid implementation into more general core/field
* field can be initialized from std::function
* pressure plugin uses region

### v0.9.4

* **fix**: stress free state can be used when the cell is grown
* **fix**: MembraneMesh wrapper needs GPU

### v0.9.3

* **fix**: the stress entries are now cleared before forces; could be cleared more by other interaction handlers
* use mirheo state inside the simulation objects; do not have current time, step and dt separate in simulation

### v0.9.2

* add filtered initial conditions: allows custom regions in space to initialise uniform density particles

### v0.9.1

* add sphere initial condition
* add plugin to compute total virial pressure from stresses per particle for a given pv

### v0.9.0

* Add common YmrState object.
* **interface change**:
  * the time step is now passed to the coordinator constructor
  * the time step is not passed to any other object

### v0.8.0

* add checkpoint for permanent channels
* extra data managers are aware of the type

### v0.7.1

* add permanent channels in extra data manager

### v0.7.0

* reorganise membrane interaction kernels:
  * bending force kernels now separated from other
  * 2 parameter strctures
* add Juelicher bending model 
* add force saver plugin to save forces in channels

### v0.6.1

* rename uDeviceX to Ymero
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

* **interface change**: wall oscillation: period is in dpd units now

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
