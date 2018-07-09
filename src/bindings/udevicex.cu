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
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>

#include <core/integrators/interface.h>


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
    sim->registerParticleVector(std::unique_ptr<ParticleVector>   (pv),
                                std::unique_ptr<InitialConditions>(ic),
                                checkpointEvery);
}
// 	void registerWall                   (PyWall wall, int checkEvery);
// 	void registerInteraction            (PyInteraction interaction);

void uDeviceX::registerIntegrator(Integrator* integrator)
{
    sim->registerIntegrator(std::unique_ptr<Integrator>(integrator));
}

// 	void uDeviceX::registerBouncer                (PyBouncer bouncer);
// 	void uDeviceX::registerPlugin                 (PyPlugin plugin);
// 	void uDeviceX::registerObjectBelongingChecker (PyObjectBelongingChecker checker);
// 
// 	void uDeviceX::setIntegrator             (std::string integratorName,  std::string pvName);
// 	void uDeviceX::setInteraction            (std::string interactionName, std::string pv1Name, std::string pv2Name);
// 	void uDeviceX::setBouncer                (std::string bouncerName,     std::string objName, std::string pvName);
// 	void uDeviceX::setWallBounce             (std::string wallName,        std::string pvName);
// 	void uDeviceX::setObjectBelongingChecker (std::string checkerName,     std::string objName);


void uDeviceX::registerPlugins( std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> > plugins )
{
	if (isComputeTask())
	{
		if ( plugins.first != nullptr && !(plugins.first->needPostproc() && noPostprocess) )
			sim->registerPlugin(std::move(plugins.first));
	}
	else
	{
		if ( plugins.second != nullptr && !noPostprocess )
			post->registerPlugin(std::move(plugins.second));
	}
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
		sim->init();
		sim->run(nsteps);
		sim->finalize();
	}
	else
		post->run();
}

