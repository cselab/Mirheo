// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mirheo.h"
#include "class_wrapper.h"

#include <mirheo/core/bouncers/interface.h>
#include <mirheo/core/initial_conditions/interface.h>
#include <mirheo/core/integrators/interface.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/object_belonging/interface.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/strprintf.h>
#include <mirheo/core/walls/interface.h>

#include <pybind11/stl.h>
#include <string>

namespace mirheo
{

using namespace pybind11::literals;

static CheckpointIdAdvanceMode getCheckpointMode(const std::string& mode)
{
    if (mode == "PingPong")
        return CheckpointIdAdvanceMode::PingPong;
    if (mode == "Incremental")
        return CheckpointIdAdvanceMode::Incremental;

    die("Unknown checkpoint mode '%s'\n", mode.c_str());
    return CheckpointIdAdvanceMode::PingPong;
}

void exportDomainInfo(py::module& m)
{
    py::class_<DomainInfo>(m, "DomainInfo", R"(
        Convert between local domain coordinates (specific to each rank) and global domain coordinates.
    )")
        .def("local_to_global", &DomainInfo::local2global, "x"_a, R"(
            Convert local coordinates to global coordinates.

            Args:
                x: Position in local coordinates.
        )")
        .def("global_to_local", &DomainInfo::global2local, "x"_a, R"(
            Convert from global coordinates to local coordinates.

            Args:
                x: Position in global coordinates.
        )")
        .def("is_in_subdomain", &DomainInfo::inSubDomain<real3>, "x"_a, R"(
            Returns True if the given position (in global coordinates) is inside the subdomain of the current rank,
            False otherwise.

            Args:
                x: Position in global coordinates.
        )")
        .def_readonly("global_size", &DomainInfo::globalSize, R"(
            Size of the whole simulation domain.
        )")
        .def_readonly("global_start", &DomainInfo::globalStart, R"(
            Subdomain lower corner position of the current rank, in global coordinates.
        )")
        .def_readonly("local_size", &DomainInfo::localSize, R"(
            Subdomain extents of the current rank.
        )")
        .def_property_readonly("local_to_global_shift", [](const DomainInfo& d) {return d.local2global(make_real3(0.0_r));}, R"(
            shift to transform local coordinates to global coordinates.
        )")
        .def_property_readonly("global_to_local_shift", [](const DomainInfo& d) {return d.global2local(make_real3(0.0_r));}, R"(
            shift to transform global coordinates to local coordinates.
        )");
}

void exportMirheo(py::module& m)
{
    py::handlers_class<MirState>(m, "MirState", R"(
        state of the simulation shared by all simulation objects.
    )")
        .def_readonly("domain_info", &MirState::domain, R"(
            The :any:`DomainInfo` of the current rank.
        )", py::return_value_policy::reference_internal)
        .def_readonly("current_time", &MirState::currentTime, R"(
            Current simulation time.
        )")
        .def_readonly("current_step", &MirState::currentStep, R"(
            Current simulation step.
        )")
        .def_property_readonly("current_dt", &MirState::getDt, R"(
            Current simulation step size dt.
            Note: this property is accessible only while Mirheo::run() is running.
        )");


    py::handlers_class<Mirheo>(m, "Mirheo", R"(
        Main coordination class, should only be one instance at a time
    )")
        .def(py::init( [] (int3 nranks, real3 domain,
                           std::string log, int debuglvl,
                           int checkpointEvery, std::string checkpointFolder, std::string checkpointModeStr,
                           real maxObjHalfLength, bool cudaMPI, bool noSplash, long commPtr)
            {
                LogInfo logInfo(log, debuglvl, noSplash);
                CheckpointInfo checkpointInfo(
                        checkpointEvery, checkpointFolder,
                        getCheckpointMode(checkpointModeStr));

                if (commPtr == 0)
                {
                    return std::make_unique<Mirheo> (nranks, domain, logInfo,
                                                     checkpointInfo, maxObjHalfLength, cudaMPI);
                }
                else
                {
                    // https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
                    MPI_Comm comm = *(MPI_Comm *)commPtr;
                    return std::make_unique<Mirheo> (comm, nranks, domain, logInfo,
                                                     checkpointInfo, maxObjHalfLength, cudaMPI);
                }
            } ),
             py::return_value_policy::take_ownership,
             "nranks"_a, "domain"_a, "log_filename"_a="log", "debug_level"_a=-1,
             "checkpoint_every"_a=0, "checkpoint_folder"_a="restart/", "checkpoint_mode"_a="PingPong",
             "max_obj_half_length"_a=0.0_r, "cuda_aware_mpi"_a=false, "no_splash"_a=false, "comm_ptr"_a=0, R"(
Create the Mirheo coordinator.

.. warning::
    Debug level determines the amount of output produced by each of the simulation processes:

        0. silent: no log output
        1. only report fatal errors
        2. report serious errors
        3. report important information steps of simulation and warnings (this is the default level)
        4. report not critical information
        5. report some debug information
        6. report more debug
        7. report all the debug
        8. force flushing to the file after each message

    Debug levels above 4 or 5 may significanlty increase the runtime, they are only recommended to debug errors.
    Flushing increases the runtime yet more, but it is required in order not to lose any messages in case of abnormal program abort.

    The default debug level may be modified by setting the ``MIRHEO_DEBUG_LEVEL`` environment variable to the desired value.
    This variable may be useful when Mirheo is linked as part of other codes, in which case the ``debug_level`` variable affects only parts of the execution.


Args:
    nranks: number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
    domain: size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries. The domain will be split in equal chunks between the MPI ranks.
        The largest chunk size that a single MPI rank can have depends on the total number of particles,
        handlers and hardware, and is typically about :math:`120^3 - 200^3`.
    log_filename: prefix of the log files that will be created.
        Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ...
        will be created, where NP is the total number of MPI ranks.
        Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all
        the individual ones and sorting with respect to time.
        If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however,
        there is no guarantee that messages from different ranks are synchronized.
    debug_level: Debug level from 0 to 8, see above.
    checkpoint_every: save state of the simulation components (particle vectors and handlers like integrators, plugins, etc.)
    checkpoint_folder: folder where the checkpoint files will reside (for Checkpoint mechanism), or folder prefix (for Snapshot mechanism)
    checkpoint_mode: set to "PingPong" to keep only the last 2 checkpoint states; set to "Incremental" to keep all checkpoint states.
    max_obj_half_length: Half of the maximum size of all objects. Needs to be set when objects are self interacting with pairwise interactions.
    cuda_aware_mpi: enable CUDA Aware MPI. The MPI library must support that feature, otherwise it may fail.
    no_splash: don't display the splash screen when at the start-up.
    comm_ptr: pointer to communicator. By default MPI_COMM_WORLD will be used
        )")

        .def("registerParticleVector", &Mirheo::registerParticleVector,
            "pv"_a, "ic"_a=nullptr, R"(
            Register particle vector

            Args:
                pv: :any:`ParticleVector`
                ic: :class:`~libmirheo.InitialConditions.InitialConditions` that will generate the initial distibution of the particles
        )")
        .def("registerIntegrator", &Mirheo::registerIntegrator,
             "integrator"_a, R"(
                Register an :any:`Integrator` to the coordinator

                Args:
                    integrator: the :any:`Integrator` to register
         )")
        .def("registerInteraction", &Mirheo::registerInteraction,
             "interaction"_a, R"(
                Register an :any:`Interaction` to the coordinator

                Args:
                    interaction: the :any:`Interaction` to register
        )")
        .def("registerObjectBelongingChecker", &Mirheo::registerObjectBelongingChecker,
             "checker"_a, "ov"_a, R"(
                Register Object Belonging Checker

                Args:
                    checker: instance of :any:`BelongingChecker`
                    ov: :any:`ObjectVector` belonging to which the **checker** will check
        )")

        .def("registerBouncer", &Mirheo::registerBouncer,
             "bouncer"_a, R"(
               Register Object Bouncer

               Args:
                   bouncer: the :any:`Bouncer` to register
        )")
        .def("registerWall", &Mirheo::registerWall,
             "wall"_a, "check_every"_a=0, R"(
               Register a :any:`Wall`.

               Args:
                   wall: the :any:`Wall` to register
                   check_every: if positive, check every this many time steps if particles penetrate the walls
        )")
        .def("registerPlugins",
             (void(Mirheo::*)(std::shared_ptr<SimulationPlugin>,
                              std::shared_ptr<PostprocessPlugin>))&Mirheo::registerPlugins,
             "Register Plugins")

        .def("deregisterIntegrator", &Mirheo::deregisterIntegrator, "integrator"_a, "Deregister a integrator.")
        .def("deregisterPlugins", &Mirheo::deregisterPlugins, "Deregister a plugin.")

        .def("setIntegrator",  &Mirheo::setIntegrator,
             "integrator"_a, "pv"_a, R"(
               Set a specific :any:`Integrator` to a given :any:`ParticleVector`

               Args:
                   integrator: the :any:`Integrator` to assign
                   pv: the concerned :any:`ParticleVector`
        )")
        .def("setInteraction", &Mirheo::setInteraction,
             "interaction"_a, "pv1"_a, "pv2"_a, R"(
                Forces between two instances of :any:`ParticleVector` (they can be the same) will be computed according to the defined interaction.

                Args:
                    interaction: :any:`Interaction` to apply
                    pv1: first :any:`ParticleVector`
                    pv2: second :any:`ParticleVector`

        )")
        .def("setBouncer", &Mirheo::setBouncer,
             "bouncer"_a, "ov"_a, "pv"_a, R"(
                Assign a :any:`Bouncer` between an :any:`ObjectVector` and a :any:`ParticleVector`.

                Args:
                    bouncer: :any:`Bouncer` compatible with the object vector
                    ov: the :any:`ObjectVector` to be bounced on
                    pv: the :any:`ParticleVector` to be bounced
        )")
        .def("setWall", &Mirheo::setWallBounce,
             "wall"_a, "pv"_a, "maximum_part_travel"_a=0.25_r, R"(
                Assign a :any:`Wall` bouncer to a given :any:`ParticleVector`.
                The current implementation does not support :any:`ObjectVector`.

                Args:
                    wall: the :any:`Wall` surface which will bounce the particles
                    pv: the :any:`ParticleVector` to be bounced
                    maximum_part_travel: maximum distance that one particle travels in one time step.
                        this should be as small as possible for performance reasons but large enough for correctness
         )")
        .def("getState", &Mirheo::getMirState, "Return mirheo state", py::return_value_policy::reference_internal)

        .def("dumpWalls2XDMF", &Mirheo::dumpWalls2XDMF,
            "walls"_a, "h"_a, "filename"_a="xdmf/wall", R"(
                Write Signed Distance Function for the intersection of the provided walls (negative values are the 'inside' of the simulation)

                Args:
                    walls: list of walls to dump; the output sdf will be the union of all walls inside
                    h: cell-size of the resulting grid
                    filename: base filename output, will create to files filename.xmf and filename.h5
        )")

        .def("computeVolumeInsideWalls", &Mirheo::computeVolumeInsideWalls,
            "walls"_a, "nSamplesPerRank"_a=100000, R"(
                Compute the volume inside the given walls in the whole domain (negative values are the 'inside' of the simulation).
                The computation is made via simple Monte-Carlo.

                Args:
                    walls: sdf based walls
                    nSamplesPerRank: number of Monte-Carlo samples used per rank
        )")

        .def("applyObjectBelongingChecker",    &Mirheo::applyObjectBelongingChecker,
            "checker"_a, "pv"_a, "correct_every"_a=0, "inside"_a="", "outside"_a="", R"(
                Apply the **checker** to the given particle vector.
                One and only one of the options **inside** or **outside** has to be specified.

                Args:
                    checker: instance of :any:`BelongingChecker`
                    pv: :any:`ParticleVector` that will be split (source PV)
                    inside: if specified and not "none", a new :any:`ParticleVector` with name **inside** will be returned, that will keep the inner particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the outer particles
                    outside: if specified and not "none", a new :any:`ParticleVector` with name **outside** will be returned, that will keep the outer particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the inner particles
                    correct_every: If greater than zero, perform correction every this many time-steps.
                        Correction will move e.g. *inner* particles of outer PV to the :inner PV
                        and viceversa. If one of the PVs was defined as "none", the 'wrong' particles will be just removed.

                Returns:
                    New :any:`ParticleVector` or None depending on **inside** and **outside** options

        )")

        .def("makeFrozenWallParticles", &Mirheo::makeFrozenWallParticles,
             "pvName"_a, "walls"_a, "interactions"_a, "integrator"_a, "number_density"_a, "mass"_a=1.0_r, "dt"_a, "nsteps"_a=1000, R"(
                Create particles frozen inside the walls.

                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead

                Args:
                    pvName: name of the created particle vector
                    walls: array of instances of :any:`Wall` for which the frozen particles will be generated
                    interactions: list of :any:`Interaction` that will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    number_density: target particle number density
                    mass: the mass of a single frozen particle
                    dt: time step
                    nsteps: run this many steps to achieve equilibrium

                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.

        )")

        .def("makeFrozenRigidParticles", &Mirheo::makeFrozenRigidParticles,
             "checker"_a, "shape"_a, "icShape"_a, "interactions"_a, "integrator"_a, "number_density"_a, "mass"_a=1.0_r, "dt"_a, "nsteps"_a=1000, R"(
                Create particles frozen inside object.

                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead

                Args:
                    checker: object belonging checker
                    shape: object vector describing the shape of the rigid object
                    icShape: initial conditions for shape
                    interactions: list of :any:`Interaction` that will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    number_density: target particle number density
                    mass: the mass of a single frozen particle
                    dt: time step
                    nsteps: run this many steps to achieve equilibrium

                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.

        )")

        .def("restart", &Mirheo::restart,
             "folder"_a="restart/", R"(
               Restart the simulation. This function should typically be called just before running the simulation.
               It will read the state of all previously registered instances of :any:`ParticleVector`, :any:`Interaction`, etc.
               If the folder contains no checkpoint file required for one of those, an error occur.

               .. warning::
                  Certain :any:`Plugins` may not implement restarting mechanism and will not restart correctly.
                  Please check the documentation for the plugins.

               Args:
                   folder: folder with the checkpoint files
        )")

        .def("isComputeTask", &Mirheo::isComputeTask, "Returns ``True`` if the current rank is a simulation task and ``False`` if it is a postrprocess task")
        .def("isMasterTask",  &Mirheo::isMasterTask,  "Returns ``True`` if the current rank is the root")
        .def("start_profiler", &Mirheo::startProfiler, "Tells nvprof to start recording timeline")
        .def("stop_profiler",  &Mirheo::stopProfiler,  "Tells nvprof to stop recording timeline")
        .def("save_dependency_graph_graphml",  &Mirheo::dumpDependencyGraphToGraphML,
             "fname"_a, "current"_a = true, R"(
             Exports `GraphML <http://graphml.graphdrawing.org/>`_ file with task graph for the current simulation time-step

             Args:
                 fname: the output filename (without extension)
                 current: if True, save the current non empty tasks; else, save all tasks that can exist in a simulation

             .. warning::
                 if current is set to True, this must be called **after** :py:meth:`mmirheo.Mirheo.run`.
         )")
        .def("run", &Mirheo::run,
             "niters"_a, "dt"_a, R"(
             Advance the system for a given amount of time steps.

             Args:
                 niters: number of time steps to advance
                 dt: time step duration
        )")
        .def("log_compile_options", &Mirheo::logCompileOptions,
             R"(
             output compile times options in the log
        )");
}

} // namespace mirheo
