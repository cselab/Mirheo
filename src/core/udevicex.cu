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

#include <core/walls/simple_stationary_wall.h>
#include <core/walls/freeze_particles.h>
#include <core/initial_conditions/uniform_ic.h>


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



void uDeviceX::registerParticleVector(const std::shared_ptr<ParticleVector>& pv, const std::shared_ptr<InitialConditions>& ic, int checkpointEvery)
{
    if (isComputeTask())
        sim->registerParticleVector(pv, ic, checkpointEvery);
}
void uDeviceX::registerIntegrator(const std::shared_ptr<Integrator>& integrator)
{
    if (isComputeTask())
        sim->registerIntegrator(integrator);
}
void uDeviceX::registerInteraction(const std::shared_ptr<Interaction>& interaction)
{
    if (isComputeTask())
        sim->registerInteraction(interaction);
}
void uDeviceX::registerWall(const std::shared_ptr<Wall>& wall, int checkEvery)
{
    if (isComputeTask())
        sim->registerWall(wall, checkEvery);
}
void uDeviceX::registerBouncer(const std::shared_ptr<Bouncer>& bouncer)
{
    if (isComputeTask())
        sim->registerBouncer(bouncer);
}
void uDeviceX::registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov)
{
    if (isComputeTask())
    {
        sim->registerObjectBelongingChecker(checker);
        sim->setObjectBelongingChecker(checker->name, ov->name);
    }
}
void uDeviceX::registerPlugins(const std::shared_ptr<SimulationPlugin>& simPlugin, const std::shared_ptr<PostprocessPlugin>& postPlugin)
{
    if (isComputeTask())
    {
        if ( simPlugin != nullptr && !(simPlugin->needPostproc() && noPostprocess) )
            sim->registerPlugin(simPlugin);
    }
    else
    {
        if ( postPlugin != nullptr && !noPostprocess )
            post->registerPlugin(postPlugin);
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

std::shared_ptr<ParticleVector> uDeviceX::makeFrozenWallParticles(std::shared_ptr<Wall> wall,
                                                                  std::shared_ptr<Interaction> interaction,
                                                                  std::shared_ptr<Integrator>   integrator,
                                                                  float density, int nsteps)
{
    if (!isComputeTask()) return nullptr;

    // Walls are not directly reusable in other simulations,
    // because they store some information like cell-lists
    //
    // But here we don't pass the wall into the other simulation,
    // we just use it to filter particles, which is totally fine
    
    info("Generating frozen particles for wall '%s'...\n\n", wall->name.c_str());
    
    auto sdfWall = dynamic_cast<SDF_basedWall*>(wall.get());
    if (sdfWall == nullptr)
        die("Only sdf-based walls are supported now!");
    
    // Check if the wall is set up
    sim->getWallByNameOrDie(wall->name);
    
    Simulation wallsim(sim->nranks3D, sim->domain.globalSize, sim->cartComm, MPI_COMM_NULL, false);
    
    auto pv=std::make_shared<ParticleVector>(wall->name, 1.0);
    auto ic=std::make_shared<UniformIC>(density);
    
    wallsim.registerParticleVector(pv, ic, 0);
    wallsim.registerInteraction(interaction);

    wallsim.registerIntegrator(integrator);
    
    wallsim.setInteraction(interaction->name, pv->name, pv->name);
    wallsim.setIntegrator (integrator->name,  pv->name);
    
    wallsim.init();
    wallsim.run(nsteps);
    
    freezeParticlesInWall(sdfWall, pv.get(), 0.0f, interaction->rc + 0.2f);
    
    return pv;
}


std::shared_ptr<ParticleVector> uDeviceX::applyObjectBelongingChecker(ObjectBelongingChecker* checker,
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
    return sim->getSharedPVbyName(newPVname);
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

