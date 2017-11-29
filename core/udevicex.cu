#include "udevicex.h"

#include <mpi.h>
#include <core/logger.h>
#include <core/simulation.h>
#include <core/postproc.h>
#include <plugins/interface.h>

uDeviceX::uDeviceX(int3 nranks3D, float3 globalDomainSize,
		Logger& logger, std::string logFileName, int verbosity)
{
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

	if      (nranks3D.x * nranks3D.y * nranks3D.z     == nranks) noPostprocess = true;
	else if (nranks3D.x * nranks3D.y * nranks3D.z * 2 == nranks) noPostprocess = false;
	else die("Asked for %d x %d x %d processes, but provided %d", nranks3D.x, nranks3D.y, nranks3D.z, nranks);

	if (rank == 0) sayHello();

	MPI_Comm ioComm, compComm, interComm, splitComm;

	if (noPostprocess)
	{
		warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");

		sim = new Simulation(nranks3D, globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL);
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

		sim = new Simulation(nranks3D, globalDomainSize, compComm, interComm);
	}
	else
	{
		MPI_Check( MPI_Comm_dup(splitComm, &ioComm) );
		MPI_Check( MPI_Intercomm_create(ioComm,   0, MPI_COMM_WORLD, 0, 0, &interComm) );

		MPI_Check( MPI_Comm_rank(ioComm, &rank) );

		post = new Postprocess(ioComm, interComm);
	}
}

void uDeviceX::registerPlugins(std::pair<SimulationPlugin*, PostprocessPlugin*> plugins)
{
	if (isComputeTask())
	{
		if ( plugins.first != nullptr && !(plugins.first->needPostproc() && noPostprocess) )
			sim->registerPlugin(plugins.first);
	}
	else
	{
		if ( plugins.second != nullptr && !noPostprocess )
			post->registerPlugin(plugins.second);
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

		CUDA_Check( cudaDeviceSynchronize() );
	}
	else
		post->run();

	MPI_Finalize();
}

