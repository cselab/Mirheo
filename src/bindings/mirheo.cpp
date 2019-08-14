#include <string>

#include <pybind11/stl.h>

#include <core/mirheo.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include "bindings.h"
#include "class_wrapper.h"

using namespace pybind11::literals;

CheckpointIdAdvanceMode getCheckpointMode(std::string mode)
{
    if (mode == "PingPong")
        return CheckpointIdAdvanceMode::PingPong;
    if (mode == "Incremental")
        return CheckpointIdAdvanceMode::Incremental;

    die("Unknown checkpoint mode '%s'\n", mode.c_str());
    return CheckpointIdAdvanceMode::PingPong;
}

void exportMirheo(py::module& m)
{
    py::handlers_class<MirState>(m, "MirState", R"(
        state of the simulation shared by all simulation objects.
    )");
    
    py::class_<Mirheo>(m, "mirheo", R"(
        Main coordination class, should only be one instance at a time
    )")
        .def(py::init( [] (PyTypes::int3 nranks, PyTypes::float3 domain, float dt,
                           std::string log, int debuglvl, int checkpointEvery,
                           std::string checkpointFolder, std::string checkpointModeStr,
                           bool cudaMPI, bool noSplash, long comm)
            {
                LogInfo logInfo(log, debuglvl, noSplash);
                auto checkpointMode = getCheckpointMode(checkpointModeStr);
                CheckpointInfo checkpointInfo(checkpointEvery, checkpointFolder, checkpointMode);
                
                if (comm == 0) return std::make_unique<Mirheo> (      nranks, domain, dt, logInfo,
                                                                      checkpointInfo, cudaMPI);
                else           return std::make_unique<Mirheo> (comm, nranks, domain, dt, logInfo,
                                                                      checkpointInfo, cudaMPI);
            } ),
            py::return_value_policy::take_ownership,
            "nranks"_a, "domain"_a, "dt"_a, "log_filename"_a="log", "debug_level"_a=3, "checkpoint_every"_a=0,
             "checkpoint_folder"_a="restart/", "checkpoint_mode"_a = "PingPong", "cuda_aware_mpi"_a=false,
             "no_splash"_a=false, "comm_ptr"_a=0, R"(
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
                
                Args:
                    nranks: number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
                    domain: size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries. The domain will be split in equal chunks between the MPI ranks.
                        The largest chunk size that a single MPI rank can have depends on the total number of particles,
                        handlers and hardware, and is typically about :math:`120^3 - 200^3`.
                    dt: timestep of the simulation
                    log_filename: prefix of the log files that will be created. 
                        Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ... 
                        will be created, where NP is the total number of MPI ranks. 
                        Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all
                        the individual ones and sorting with respect to time.
                        If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however, 
                        there is no guarantee that messages from different ranks are synchronized.
                    debug_level: Debug level from 0 to 8, see above.
                    checkpoint_every: save state of the simulation components (particle vectors and handlers like integrators, plugins, etc.)
                    checkpoint_folder: folder where the checkpoint files will reside
                    checkpoint_mode: set to "PingPong" to keep only the last 2 checkpoint states; set to "Incremental" to keep all checkpoint states.
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
        
        .def("registerBouncer",  &Mirheo::registerBouncer,
             "bouncer"_a, R"(
               Register Object Bouncer

               Args:
                   bouncer: the :any:`Bouncer` to register
        )")
        .def("registerWall",     &Mirheo::registerWall,
             "wall"_a, "check_every"_a=0, R"(
               Register a :any:`Wall`.

               Args:
                   wall: the :any:`Wall` to register
                   check_every: if positive, check every this many time steps if particles penetrate the walls 
        )")
        .def("registerPlugins",  &Mirheo::registerPlugins, "Register Plugins")

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
        .def("setWall",        &Mirheo::setWallBounce,
             "wall"_a, "pv"_a, "maximum_part_travel"_a=0.25f, R"(
                Assign a :any:`Wall` bouncer to a given :any:`ParticleVector`.
                The current implementation does not support :any:`ObjectVector`.

                Args:
                    wall: the :any:`Wall` surface which will bounce the particles
                    pv: the :any:`ParticleVector` to be bounced
                    maximum_part_travel: maximum distance that one particle travels in one time step.
                        this should be as small as possible for performance reasons but large enough for correctness
         )")
        .def("getState",       &Mirheo::getMirState,    "Return mirheo state")
        
        .def("dumpWalls2XDMF",    &Mirheo::dumpWalls2XDMF,
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
             "pvName"_a, "walls"_a, "interactions"_a, "integrator"_a, "density"_a, "nsteps"_a=1000, R"(
                Create particles frozen inside the walls.
                
                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead
                
                Args:
                    pvName: name of the created particle vector
                    walls: array of instances of :any:`Wall` for which the frozen particles will be generated
                    interactions: list of :any:`Interaction` that will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    density: target particle density
                    nsteps: run this many steps to achieve equilibrium
                            
                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.
                    
        )")

        .def("makeFrozenRigidParticles", &Mirheo::makeFrozenRigidParticles,
             "checker"_a, "shape"_a, "icShape"_a, "interactions"_a, "integrator"_a, "density"_a, "nsteps"_a=1000, R"(
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
                    density: target particle density
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
        .def("save_dependency_graph_graphml",  &Mirheo::saveDependencyGraph_GraphML,
             "fname"_a, "current"_a = true, R"(
             Exports `GraphML <http://graphml.graphdrawing.org/>`_ file with task graph for the current simulation time-step
             
             Args:
                 fname: the output filename (without extension)
                 current: if True, save the current non empty tasks; else, save all tasks that can exist in a simulation
             
             .. warning::
                 if current is set to True, this must be called **after** :py:meth:`_mirheo.mirheo.run`.
         )")
        .def("run", &Mirheo::run,
             "niters"_a, R"(
             Advance the system for a given amount of time steps.

             Args:
                 niters: number of time steps to advance
        )");
}
