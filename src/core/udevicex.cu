#include "udevicex.h"

#include <mpi.h>
#include <core/logger.h>
#include <core/simulation.h>
#include <core/postproc.h>
#include <plugins/interface.h>

#include <core/utils/make_unique.h>
#include <core/utils/cuda_common.h>
#include <cuda_runtime.h>

#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>


uDeviceX::uDeviceX(std::tuple<int, int, int> nranks3D, std::tuple<float, float, float> globalDomainSize,
        std::string logFileName, int verbosity, bool gpuAwareMPI)
{
    int3 _nranks3D = make_int3(nranks3D);
    float3 _globalDomainSize = make_float3(globalDomainSize);
    
    MPI_Init(nullptr, nullptr);
    
    int nranks, rank;

    if (logFileName == "stdout")
        logger.init(MPI_COMM_WORLD, stdout, verbosity);
    else if (logFileName == "stderr")
        logger.init(MPI_COMM_WORLD, stderr, verbosity);
    else
        logger.init(MPI_COMM_WORLD, logFileName+".log", verbosity);

    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

    if      (_nranks3D.x * _nranks3D.y * _nranks3D.z     == nranks) noPostprocess = true;
    else if (_nranks3D.x * _nranks3D.y * _nranks3D.z * 2 == nranks) noPostprocess = false;
    else die("Asked for %d x %d x %d processes, but provided %d", _nranks3D.x, _nranks3D.y, _nranks3D.z, nranks);

    if (rank == 0) sayHello();

    MPI_Comm ioComm, compComm, interComm, splitComm;

    if (noPostprocess)
    {
        warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");

        sim = std::make_unique<Simulation> (_nranks3D, _globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL, gpuAwareMPI);
        computeTask = 0;
        return;
    }

    info("Program started, splitting communicator");

    computeTask = (rank) % 2;
    MPI_Check( MPI_Comm_split(MPI_COMM_WORLD, computeTask, rank, &splitComm) );

    if (isComputeTask())
    {
        MPI_Check( MPI_Comm_dup(splitComm, &compComm) );
        MPI_Check( MPI_Intercomm_create(compComm, 0, MPI_COMM_WORLD, 1, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(compComm, &rank) );

        sim = std::make_unique<Simulation> (_nranks3D, _globalDomainSize, compComm, interComm, gpuAwareMPI);
    }
    else
    {
        MPI_Check( MPI_Comm_dup(splitComm, &ioComm) );
        MPI_Check( MPI_Intercomm_create(ioComm,   0, MPI_COMM_WORLD, 0, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(ioComm, &rank) );

        post = std::make_unique<Postprocess> (ioComm, interComm);
    }
}

uDeviceX::~uDeviceX() = default;



void uDeviceX::registerParticleVector(ParticleVector* pv, InitialConditions* ic, int checkpointEvery)
{
    if (isComputeTask())
        sim->registerParticleVector(std::unique_ptr<ParticleVector>   (pv),
                                    std::unique_ptr<InitialConditions>(ic),
                                    checkpointEvery);
}
void uDeviceX::registerIntegrator(Integrator* integrator)
{
    if (isComputeTask())
        sim->registerIntegrator(std::unique_ptr<Integrator>(integrator));
}
void uDeviceX::registerInteraction(Interaction* interaction)
{
    if (isComputeTask())
        sim->registerInteraction(std::unique_ptr<Interaction>(interaction));
}
void uDeviceX::registerWall(Wall* wall, int checkEvery)
{
    if (isComputeTask())
        sim->registerWall(std::unique_ptr<Wall>(wall), checkEvery);
}
void uDeviceX::registerBouncer(Bouncer* bouncer)
{
    if (isComputeTask())
        sim->registerBouncer(std::unique_ptr<Bouncer>(bouncer));
}
void uDeviceX::registerObjectBelongingChecker (ObjectBelongingChecker* checker, ObjectVector* ov)
{
    if (isComputeTask())
    {
        sim->registerObjectBelongingChecker(std::unique_ptr<ObjectBelongingChecker>(checker));
        sim->setObjectBelongingChecker(checker->name, ov->name);
    }
}
void uDeviceX::registerPlugins(SimulationPlugin* simPlugin, PostprocessPlugin* postPlugin)
{
    if (isComputeTask())
    {
        if ( simPlugin != nullptr && !(simPlugin->needPostproc() && noPostprocess) )
            sim->registerPlugin(std::unique_ptr<SimulationPlugin>(simPlugin));
    }
    else
    {
        if ( postPlugin != nullptr && !noPostprocess )
            post->registerPlugin(std::unique_ptr<PostprocessPlugin>(postPlugin));
    }
}

void uDeviceX::setIntegrator(Integrator* integrator, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setIntegrator(integrator->name, pv->name);
}
void uDeviceX::setInteraction(Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2)
{
    if (isComputeTask())
        sim->setInteraction(interaction->name, pv1->name, pv2->name);
}
void uDeviceX::setBouncer(Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setBouncer(bouncer->name, ov->name, pv->name);
}
void uDeviceX::setWallBounce(Wall* wall, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setWallBounce(wall->name, pv->name);
}

ParticleVector* uDeviceX::applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                ParticleVector* pv,
                                                int checkEvery,
                                                std::string inside,
                                                std::string outside)
{
    if (!isComputeTask()) return nullptr;
    
    if ( (inside != "" && outside != "") || (inside == "" && outside == "") )
        die("One and only one option can be specified for belonging checker '%s': inside or outside",
            checker->name.c_str());
    
    std::string newPVname;
    
    if (inside == "")
    {
        inside = pv->name;
        newPVname = outside;
    }
    if (outside == "")
    {
        outside = pv->name;
        newPVname = inside;
    }
        
    sim->applyObjectBelongingChecker(checker->name, pv->name, inside, outside, checkEvery);
    return sim->getPVbyName(newPVname);
}


void uDeviceX::sayHello()
{
    printf("\n");
    printf("************************************************\n");
    printf("*                   uDeviceX                   *\n");
    printf("*     compiled: on %s at %s     *\n", __DATE__, __TIME__);
    printf("************************************************\n");
    printf("\n");
}

bool uDeviceX::isComputeTask()
{
    return computeTask == 0;
}

void uDeviceX::run(int nsteps)
{
    if (isComputeTask())
    {
        sim->init();  // TODO reentrant!!
        sim->run(nsteps);
        sim->finalize();
    }
    else
        post->run();
}

