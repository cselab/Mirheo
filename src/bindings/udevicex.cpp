#include <pybind11/pybind11.h>

#include <core/udevicex.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportUdevicex(py::module& m)
{
    py::class_<uDeviceX>(m, "udevicex", R"(
        Main coordination class, should only be one instance at a time
    )")
        .def(py::init< pyint3, pyfloat3, std::string, int, bool >(),
             "nranks"_a, "domain"_a, "log_filename"_a="log", "debug_level"_a=3, "cuda_aware_mpi"_a=false, R"(
            Args:
                nranks:
                    number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
                domain:
                    size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries.
                    The domain will be split in equal chunks between the MPI ranks.
                    The largest chunk size that a single MPI rank can have depends on the total number of particles,
                    handlers and hardware, and is typically about :math:`120^3 - 200^3`.
                log_filename:
                    prefix of the log files that will be created. Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ... will be created, where NP is the total number of MPI ranks. Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all the individual ones and sorting with respect to time.
                    If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however, there is no guarantee that messages from different ranks are synchronized
                debug_level:
                    | Debug level varies from 1 to 8:
                    | 
                    | #. only report fatal errors
                    | #. report serious errors
                    | #. report warnings (this is the default level)
                    | #. report not critical information
                    | #. report some debug information
                    | #. report more debug
                    | #. report all the debug
                    | #. force flushing to the file after each message
                    | 
                    | Debug levels above 4 or 5 may significanlty increase the runtime, they are only recommended to debug errors.
                    | Flushing increases the runtime yet more, but it is required in order not to lose any messages in case of abnormal program abort.
                cuda_aware_mpi: enable CUDA Aware MPI (GPU RDMA). As of now it may crash, or may yield slower execution.
        )")
        
        .def("registerParticleVector", &uDeviceX::registerParticleVector,
            "pv"_a, "ic"_a, "checkpoint_every"_a=0, R"(
            Register particle vector
            
            Args:
                pv: :any:`ParticleVector`
                ic: :class:`~libudevicex.InitialConditions.InitialConditions` that will generate the initial distibution of the particles
                checkpoint_every:
                    every that many timesteps the state of the Particle Vector across all the MPI processes will be saved to disk  into the ./restart/ folder. The checkpoint files may be used to restart the whole simulation or only some individual PVs from the saved states. Default value of 0 means no checkpoint.
        )")
        .def("registerIntegrator",             &uDeviceX::registerIntegrator,             "Register Integrator")
        .def("registerInteraction",            &uDeviceX::registerInteraction,            "Register Interaction")
        .def("registerObjectBelongingChecker", &uDeviceX::registerObjectBelongingChecker,
             "checker"_a, "ov"_a, R"(
                Register Object Belonging Checker
                
                Args:
                    checker: instance of :any:`BelongingChecker`
                    ov: :any:`ObjectVector` belonging to which the **checker** will check
        )")
        
        .def("registerBouncer",                &uDeviceX::registerBouncer,                "Register Object Bouncer")
        .def("registerWall",                   &uDeviceX::registerWall,                   "Register Wall")
        .def("registerPlugins",                &uDeviceX::registerPlugins,                "Register Plugins")

        .def("setIntegrator",  &uDeviceX::setIntegrator,  "Set Integrator")
        .def("setInteraction", &uDeviceX::setInteraction,
             "interaction"_a, "pv1"_a, "pv2"_a, R"(
                Forces between two Particle Vectors (they can be the same) *name1* and *name2* will be computed according to the defined interaction.
        )")
        .def("setBouncer",     &uDeviceX::setBouncer,     "Set Bouncer")
        .def("setWall",        &uDeviceX::setWallBounce,  "Set Wall")
        .def("applyObjectBelongingChecker",    &uDeviceX::applyObjectBelongingChecker,
            "checker"_a, "pv"_a, "correct_every"_a=0, "inside"_a="", "outside"_a="", R"(
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
                            
                Returns:
                    New :any:`ParticleVector` or None depending on **inside** and **outside** options
                    
        )")
        
        .def("isComputeTask", &uDeviceX::isComputeTask, "Returns whether current rank will do compute or postrprocess")
        .def("run", &uDeviceX::run, "Run the simulation");
}
