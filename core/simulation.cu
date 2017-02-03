#include <core/simulation.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/redistributor.h>
#include <core/halo_exchanger.h>
#include <core/logger.h>

Simulation::Simulation(int3 nranks3D, float3 globalDomainSize, MPI_Comm& comm, MPI_Comm& interComm) :
nranks3D(nranks3D), globalDomainSize(globalDomainSize), interComm(interComm)
{
	int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
	int periods[] = {1, 1, 1};
	int coords[3];

	MPI_Check( MPI_Comm_rank(comm, &rank) );
	MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
	MPI_Check( MPI_Cart_get(cartComm, 3, ranksArr, periods, coords) );
	rank3D = {coords[0], coords[1], coords[2]};

	subDomainSize = globalDomainSize / make_float3(nranks3D);
	subDomainStart = {subDomainSize.x * coords[0], subDomainSize.y * coords[1], subDomainSize.y * coords[2]};
}

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;
	particleVectors.push_back(pv);

	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvMap[name] = particleVectors.size() - 1;
	ic->exec(pv, globalDomainSize, subDomainSize);
}

void Simulation::registerObjectVector  (ObjectVector* ov)
{
	std::string name = ov->name;
	particleVectors.push_back(static_cast<ParticleVector*>(ov));

	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvMap[name] = particleVectors.size() - 1;
}

void Simulation::registerWall          (Wall* wall)
{
	std::string name = wall->name;

	if (wallMap.find(name) != wallMap.end())
		die("More than one wall is called %s", name.c_str());

	if (pvMap.find(name) != pvMap.end())
		die("Wall has the same name as particle vector: %s", name.c_str());

	wallMap[name] = wall;

	particleVectors.push_back(wall->getFrozen());
	pvMap[name] = particleVectors.size() - 1;
}

void Simulation::registerInteraction   (Interaction* interaction)
{
	std::string name = interaction->name;
	if (interactionMap.find(name) != interactionMap.end())
		die("More than one interaction is called %s", name.c_str());

	interactionMap[name] = interaction;
}

void Simulation::registerIntegrator    (Integrator* integrator)
{
	std::string name = integrator->name;
	if (integratorMap.find(name) != integratorMap.end())
		die("More than one interaction is called %s", name.c_str());

	integratorMap[name] = integrator;
}

void Simulation::registerPlugin(SimulationPlugin* plugin)
{
	plugins.push_back(plugin);
}

void Simulation::setIntegrator(std::string pvName, std::string integratorName)
{
	if (pvMap.find(pvName) == pvMap.end())
		die("No such particle vector: %s", pvName.c_str());

	if (integratorMap.find(integratorName) == integratorMap.end())
		die("No such integrator: %s", integratorName.c_str());

	const int pvId = pvMap[pvName];
	integrators.resize(std::max((int)integrators.size(), pvId+1), nullptr);
	integrators[pvId] = integratorMap[integratorName];
}

void Simulation::setInteraction(std::string pv1Name, std::string pv2Name, std::string interactionName)
{
	if (pvMap.find(pv1Name) == pvMap.end())
		die("No such particle vector: %s", pv1Name.c_str());

	if (pvMap.find(pv2Name) == pvMap.end())
		die("No such particle vector: %s", pv2Name.c_str());

	if (interactionMap.find(interactionName) == interactionMap.end())
		die("No such integrator: %s", interactionName.c_str());

	const int pv1Id = pvMap[pv1Name];
	const int pv2Id = pvMap[pv2Name];

	// Allocate interactionTable
	interactionTable.resize(std::max((int)interactionTable.size(), pv1Id+1));
	auto& interactionVector = interactionTable[pv1Id];
	interactionVector.resize( std::max((int)interactionVector.size(), pv2Id+1), {nullptr, nullptr} );

	// Find interaction
	auto interaction = interactionMap[interactionName];

	cellListTable.resize(std::max((int)cellListTable.size(), pv1Id+1));

	CellList* cl = nullptr;
	for (auto& entry : cellListTable[pv1Id])
	{
		if (fabs(entry->rc - interaction->rc) < 1e-6)
		{
			cl = entry;
			break;
		}
	}
	if (cl == nullptr)
	{
		cl = new CellList(particleVectors[pv1Id], interaction->rc, subDomainStart, subDomainSize);
		cellListTable[pv1Id].push_back(cl);
	}

	interactionTable[pv1Id][pv2Id] = {interaction, cl};
}

// TODO: wall has self-interactions
void Simulation::run(int nsteps)
{
	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	for (auto& pl : plugins)
	{
		pl->setup(this, defStream, cartComm, interComm);
		pl->handshake();
	}

	// TODO: STREAMS FOR CELL-LISTS

	HaloExchanger halo(cartComm);
	Redistributor redist(cartComm);

	cellListTable.resize(particleVectors.size());
	for (int i=0; i<particleVectors.size(); i++)
	{
		if (cellListTable[i].size() > 0)
		{
			auto it = std::max_element(cellListTable[i].begin(), cellListTable[i].end(),
					[] (CellList* cl1, CellList* cl2) { return cl1->rc < cl2->rc; } );
			halo.attach(particleVectors[i], *it);
			redist.attach(particleVectors[i], *it);
		}

		particleVectors[i]->setStream(defStream);
		for (auto& cl : cellListTable[i])
			cl->setStream(defStream);
	}

	float t = 0;
	for (int iter=0; iter<nsteps; iter++)
	{
		//===================================================================================================
		for (auto& pv : particleVectors)
			pv->forces.clear();

		//===================================================================================================
		for (auto& cllist : cellListTable)
			for (auto& cl : cllist)
				cl->build(defStream);

		for (auto& pl : plugins)
			pl->beforeForces(t);

		//===================================================================================================
		for (int i=0; i<interactionTable.size(); i++)
			for (int j=0; j<interactionTable[i].size(); j++)
				if (interactionTable[i][j].first != nullptr)
				{
					if (i == j)
					{
						if (interactionTable[i][j].first != nullptr)
							interactionTable[i][j].first->execSelf(particleVectors[i], interactionTable[i][j].second, t, defStream);
					}
					else
					{
						if (interactionTable[i][j].first != nullptr)
							interactionTable[i][j].first->execExternal(particleVectors[i], particleVectors[j], interactionTable[i][j].second, t, defStream);
					}
				}

		//===================================================================================================
		halo.exchange();

		//===================================================================================================
		for (int i=0; i<interactionTable.size(); i++)
			for (int j=0; j<interactionTable[i].size(); j++)
				if (interactionTable[i][j].first != nullptr)
				{
					if (interactionTable[i][j].first != nullptr)
						interactionTable[i][j].first->execHalo(particleVectors[i], particleVectors[j], interactionTable[i][j].second, t, defStream);
				}

		for (auto& pl : plugins)
			pl->beforeIntegration(t);

		//===================================================================================================
		for (int i=0; i<integrators.size(); i++)
			if (integrators[i] != nullptr)
				integrators[i]->exec(particleVectors[i], defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );

		for (auto& pl : plugins)
			pl->afterIntegration(t);

		//===================================================================================================
		redist.redistribute();
		CUDA_Check( cudaStreamSynchronize(defStream) );
	}

	MPI_Check( MPI_Barrier(cartComm) );
}


//===================================================================================================
// Postprocessing
//===================================================================================================

Postprocess::Postprocess(MPI_Comm& comm, MPI_Comm& interComm) : comm(comm), interComm(interComm) {};

void Postprocess::registerPlugin(PostprocessPlugin* plugin)
{
	plugins.push_back(plugin);
}

void Postprocess::run()
{
	for (auto& pl : plugins)
	{
		pl->setup(comm, interComm);
		pl->handshake();
	}

	std::vector<MPI_Request> requests;
	for (auto& pl : plugins)
		requests.push_back(pl->postRecv());

	// TODO: need stopping criterion
	while (true)
	{
		int index;
		MPI_Check( MPI_Waitany(requests.size(), requests.data(), &index, MPI_STATUS_IGNORE) );

		plugins[index]->deserialize();
		requests[index] = plugins[index]->postRecv();
	}
}



//===================================================================================================
// uDeviceX
//===================================================================================================

uDeviceX::uDeviceX(int argc, char** argv, int3 nranks3D, float3 globalDomainSize, Logger& logger, std::string logFileName, int verbosity)
{
	int nranks, rank;

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, logFileName, verbosity);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

	MPI_Comm ioComm, compComm, interComm, splitComm;

	if (nranks % 2 != 0)
		die("Number of MPI ranks should be even");

	debug("Program started, splitting commuticator");

	computeTask = (rank+1) % 2;
	MPI_Check( MPI_Comm_split(MPI_COMM_WORLD, computeTask, rank, &splitComm) );

	if (isComputeTask())
	{
		MPI_Check( MPI_Comm_dup(splitComm, &compComm) );
		MPI_Check( MPI_Intercomm_create(compComm, 0, MPI_COMM_WORLD, 0, 0, &interComm) );

		sim = new Simulation(nranks3D, globalDomainSize, compComm, interComm);
	}
	else
	{
		MPI_Check( MPI_Comm_dup(splitComm, &ioComm) );
		MPI_Check( MPI_Intercomm_create(ioComm,   0, MPI_COMM_WORLD, 1, 0, &interComm) );

		post = new Postprocess(ioComm, interComm);
	}
}

bool uDeviceX::isComputeTask()
{
	return computeTask == 0;
}

void uDeviceX::registerJointPlugins(SimulationPlugin* simPl, PostprocessPlugin* postPl)
{
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

void uDeviceX::run()
{
	if (computeTask)
		sim->run(100000);
	else
		post->run();

	if (computeTask)
	{
		CUDA_Check( cudaDeviceSynchronize() );
		CUDA_Check( cudaDeviceReset() );
	}

	MPI_Check( MPI_Finalize() );
}




