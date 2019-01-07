#include <string>

#include <pybind11/stl.h>

#include <core/ymero.h>
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

void exportYmero(py::module& m)
{
    py::handlers_class<YmrState>(m, "YmrState", R"(
        state of the simulation shared by all simulation objects.
    )");
    
    py::class_<YMeRo>(m, "ymero", R"(
        Main coordination class, should only be one instance at a time
    )")
        .def(py::init< PyTypes::int3, PyTypes::float3, float, std::string, int, int, std::string, bool, bool >(),
             py::return_value_policy::take_ownership,
             "nranks"_a, "domain"_a, "dt"_a, "log_filename"_a="log", "debug_level"_a=3, "checkpoint_every"_a=0, 
             "restart_folder"_a="restart/", "cuda_aware_mpi"_a=false, "no_splash"_a=false, R"(
            Args:
                nranks:
                    number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
                domain:
                    size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries.
                    The domain will be split in equal chunks between the MPI ranks.
                    The largest chunk size that a single MPI rank can have depends on the total number of particles,
                    handlers and hardware, and is typically about :math:`120^3 - 200^3`.
                dt:
                    timestep of the simulation
                log_filename:
                    prefix of the log files that will be created. 
                    Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ... will be created, where NP is the total number of MPI ranks. 
                    Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all the individual ones and sorting with respect to time.
                    If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however, there is no guarantee that messages from different ranks are synchronized
                debug_level:
                    Debug level varies from 1 to 8:
                    
                       #. only report fatal errors
                       #. report serious errors
                       #. report warnings (this is the default level)
                       #. report not critical information
                       #. report some debug information
                       #. report more debug
                       #. report all the debug
                       #. force flushing to the file after each message
                    
                    Debug levels above 4 or 5 may significanlty increase the runtime, they are only recommended to debug errors.
                    Flushing increases the runtime yet more, but it is required in order not to lose any messages in case of abnormal program abort.
                checkpoint_every:
                    save state of the simulation components (particle vectors and handlers like integrators, plugins, etc.)
                restart_folder:
                    folder where the checkpoint files will reside
                cuda_aware_mpi: enable CUDA Aware MPI (GPU RDMA). As of now it may crash, or may yield slower execution.
                no_splash: Don't display the splash screen when at the start-up.
                
        )")

        .def(py::init< long, PyTypes::int3, PyTypes::float3, float, std::string, int, int, std::string, bool >(),
             py::return_value_policy::take_ownership,
             "commPtr"_a, "nranks"_a, "domain"_a, "dt"_a, "log_filename"_a="log", "debug_level"_a=3, 
             "checkpoint_every"_a=0, "restart_folder"_a="restart/", "cuda_aware_mpi"_a=false, R"(
            Args:
                commPtr: pointer to communicator
        )")
        
        .def("registerParticleVector", &YMeRo::registerParticleVector,
            "pv"_a, "ic"_a=nullptr, "checkpoint_every"_a=0, R"(
            Register particle vector
            
            Args:
                pv: :any:`ParticleVector`
                ic: :class:`~libymero.InitialConditions.InitialConditions` that will generate the initial distibution of the particles
                checkpoint_every:
                    every that many timesteps the state of the Particle Vector across all the MPI processes will be saved to disk  into the ./restart/ folder. The checkpoint files may be used to restart the whole simulation or only some individual PVs from the saved states. Default value of 0 means no checkpoint.
        )")
        .def("registerIntegrator",             &YMeRo::registerIntegrator,             "Register Integrator")
        .def("registerInteraction",            &YMeRo::registerInteraction,            "Register Interaction")
        .def("registerObjectBelongingChecker", &YMeRo::registerObjectBelongingChecker,
             "checker"_a, "ov"_a, R"(
                Register Object Belonging Checker
                
                Args:
                    checker: instance of :any:`BelongingChecker`
                    ov: :any:`ObjectVector` belonging to which the **checker** will check
        )")
        
        .def("registerBouncer",  &YMeRo::registerBouncer, "Register Object Bouncer")
        .def("registerWall",     &YMeRo::registerWall, "wall"_a, "check_every"_a=0, "Register Wall")
        .def("registerPlugins",  &YMeRo::registerPlugins, "Register Plugins")

        .def("setIntegrator",  &YMeRo::setIntegrator,  "Set Integrator")
        .def("setInteraction", &YMeRo::setInteraction,
             "interaction"_a, "pv1"_a, "pv2"_a, R"(
                Forces between two Particle Vectors (they can be the same) *name1* and *name2* will be computed according to the defined interaction.
        )")
        .def("setBouncer",     &YMeRo::setBouncer,     "Set Bouncer")
        .def("setWall",        &YMeRo::setWallBounce,  "Set Wall")

        .def("getState",       &YMeRo::getYmrState,    "Return ymero state")
        
        .def("dumpWalls2XDMF",    &YMeRo::dumpWalls2XDMF,
            "walls"_a, "h"_a, "filename"_a="xdmf/wall", R"(
                Write Signed Distance Function for the intersection of the provided walls (negative values are the 'inside' of the simulation)
                
                Args:
                    h: cell-size of the resulting grid                    
        )")        

        .def("computeVolumeInsideWalls", &YMeRo::computeVolumeInsideWalls,
            "walls"_a, "nSamplesPerRank"_a=100000, R"(
                Compute the volume inside the given walls in the whole domain (negative values are the 'inside' of the simulation).
                The computation is made via simple Monte-Carlo.
                
                Args:
                    walls: sdf based walls
                    nSamplesPerRank: number of Monte-Carlo samples used per rank
        )")        
        
        .def("applyObjectBelongingChecker",    &YMeRo::applyObjectBelongingChecker,
            "checker"_a, "pv"_a, "correct_every"_a=0, "inside"_a="", "outside"_a="", "checkpoint_every"_a=0, R"(
                Apply the **checker** to the given particle vector.
                One and only one of the options **inside** or **outside** has to be specified.
                
                Args:
                    checker: instance of :any:`BelongingChecker`
                    pv: :any:`ParticleVector` that will be split (source PV) 
                    inside:
                        if specified and not "none", a new :any:`ParticleVector` with name **inside** will be returned, that will keep the inner particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the outer particles
                    outside:
                        if specified and not "none", a new :any:`ParticleVector` with name **outside** will be returned, that will keep the outer particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the inner particles
                    correct_every:
                        If greater than zero, perform correction every this many time-steps.                        
                        Correction will move e.g. *inner* particles of outer PV to the :inner PV
                        and viceversa. If one of the PVs was defined as "none", the 'wrong' particles will be just removed.
                    checkpoint_every:
                        every that many timesteps the state of the newly created :any:`ParticleVector` (if any) will be saved to disk into the ./restart/ folder. Default value of 0 means no checkpoint.
                            
                Returns:
                    New :any:`ParticleVector` or None depending on **inside** and **outside** options
                    
        )")
        
        .def("makeFrozenWallParticles", &YMeRo::makeFrozenWallParticles,
             "pvName"_a, "walls"_a, "interaction"_a, "integrator"_a, "density"_a, "nsteps"_a=1000, R"(
                Create particles frozen inside the walls.
                
                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead
                
                Args:
                    pvName: name of the created particle vector
                    walls: array of instances of :any:`Wall` for which the frozen particles will be generated
                    interaction: this :any:`Interaction` will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    density: target particle density
                    nsteps: run this many steps to achieve equilibrium
                            
                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.
                    
        )")

        .def("makeFrozenRigidParticles", &YMeRo::makeFrozenRigidParticles,
             "checker"_a, "shape"_a, "icShape"_a, "interaction"_a, "integrator"_a, "density"_a, "nsteps"_a=1000, R"(
                Create particles frozen inside object.
                
                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead
                
                Args:
                    checker: object belonging checker
                    shape: object vector describing the shape of the rigid object
                    icShape: initial conditions for shape
                    interaction: this :any:`Interaction` will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    density: target particle density
                    nsteps: run this many steps to achieve equilibrium
                            
                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.
                    
        )")
        
        .def("restart", &YMeRo::restart, "Restart the simulation")
        .def("isComputeTask", &YMeRo::isComputeTask, "Returns whether current rank will do compute or postrprocess")
        .def("isMasterTask",  &YMeRo::isMasterTask,  "Returns whether current task is the very first one")
        .def("start_profiler", &YMeRo::startProfiler, "Tells nvprof to start recording timeline")
        .def("stop_profiler",  &YMeRo::stopProfiler,  "Tells nvprof to stop recording timeline")
        .def("save_dependency_graph_graphml",  &YMeRo::saveDependencyGraph_GraphML,  R"(
            Exports `GraphML <http://graphml.graphdrawing.org/>`_ file with task graph for the current simulation time-step)")
        .def("run", &YMeRo::run, "Run the simulation");
}
