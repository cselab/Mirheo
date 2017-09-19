#include "udevicex.h"

#include <mpi.h>
#include <core/logger.h>

uDeviceX::uDeviceX(int argc, char** argv, int3 nranks3D, float3 globalDomainSize,
		Logger& logger, std::string logFileName, int verbosity, bool noPostprocess) : noPostprocess(noPostprocess)
{
	int nranks, rank;

	MPI_Init(&argc, &argv);

	logger.init(MPI_COMM_WORLD, logFileName, verbosity);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

	MPI_Comm ioComm, compComm, interComm, splitComm;

	if (noPostprocess)
	{
		warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");

		sim = new Simulation(nranks3D, globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL);
		computeTask = 0;
		return;
	}

	if (nranks % 2 != 0)
		die("Number of MPI ranks should be even");

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

bool uDeviceX::isComputeTask()
{
	return computeTask == 0;
}

void uDeviceX::registerJointPlugins(SimulationPlugin* simPl, PostprocessPlugin* postPl)
{
	if (noPostprocess) return;

	const int id = pluginId++;

	if (isComputeTask())
	{
		simPl->setId(id);
		sim->registerPlugin(simPl);
	}
	else
	{
		postPl->setId(id);
		post->registerPlugin(postPl);
	}
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

