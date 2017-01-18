#include "simulation.h"

Simulation::Simulation(int3 nranks, float3 fullDomainSize, MPI_Comm& comm) :
nranks(nranks), fullDomainSize(fullDomainSize)
{
	int ranksArr[] = {nranks.x, nranks.y, nranks.z};
	int periods[] = {1, 1, 1};
	int coords[3];

	MPI_Check( MPI_Comm_rank(comm, &rank) );
	MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
	MPI_Check( MPI_Cart_get(cartComm, 3, ranksArr, periods, coords) );
	rank3D = {coords[0], coords[1], coords[2]};

	subDomainSize = fullDomainSize / nranks;
	subDomainStart = subDomainSize / 2.0f;
}

void Simulation::registerParticleVector(std::string name, ParticleVector* pv)
{
	particleVectors.push_back(pv);

	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name);

	pvMap[name] = particleVectors.size() - 1;
}

void Simulation::registerObjectVector  (std::string name, ObjectVector* ov)
{
	particleVectors.push_back(static_cast<ParticleVector*>(pv));

	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name);

	pvMap[name] = particleVectors.size() - 1;
}

void Simulation::registerInteraction   (std::string name, Interaction* interaction)
{
	if (interactionMap.find(name) != interactionMap.end())
		die("More than one interaction is called %s", name);

	interactionMap[name] = interaction;
}

void Simulation::registerIntegrator    (std::string name, Integrator* integrator)
{
	if (integratorMap.find(name) != integratorMap.end())
		die("More than one interaction is called %s", name);

	integratorMap[name] = integrator;
}

void Simulation::registerWall          (std::string name, Wall* wall)
{
	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name);

	integrators[name] = integrator;
}

void Simulation::registerPlugin(SimulationPlugin* plugin)
{
	plugins.push_back(plugin);
}

void Simulation::setIntegrator(std::string pvName, std::string integratorName)
{
	if (pvMap.find(pvName) == pvMap.end())
		die("No such particle vector: %s", pvName);

	if (integratorMap.find(integratorName) == integratorMap.end())
		die("No such integrator: %s", integratorName);

	const int pvId = pvMap[pvName];
	integrators.resize(std::max(integrators.size(), pvId+1), nullptr);
	integrators[pvId] = integratorMap[integratorName];
}

void Simulation::setInteraction(std::string pv1Name, std::string pv2Name, std::string integratorName)
{
	if (pvMap.find(pv1Name) == pvMap.end())
		die("No such particle vector: %s", pv1Name);

	if (pvMap.find(pv2Name) == pvMap.end())
		die("No such particle vector: %s", pv2Name);

	if (integractionMap.find(integrationName) == integractionMap.end())
		die("No such integrator: %s", integrationName);

	const int pv1Id = pvMap[pv1Name];
	const int pv2Id = pvMap[pv2Name];

	// Allocate interactionTable
	interactionTable.resize(std::max(interactionTable.size(), pv1Id+1));
	auto& interactionVector = interactionTable[pv1Id];
	interactionVector.resize( std::max(interactionVector.size(), pv2Id+1), {nullptr, nullptr} );

	// Find interaction
	auto interaction = interactionMap[interactionName];

	cellListTable.resize(std::max(cellListTable.size(), pv1Id+1));

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
		cl = new CellList(particleVectors[pv1Id], interaction->rc, subDomainStart, subDomainSize);

	interactionTable[pv1Id][pv2Id] = {interaction, cl};
}

//
//void Simulation::prepare(MPI_Comm& comm)
//{
//
//}


void Simulation::run(int nsteps)
{
	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	for (auto& p : plugins)
		p->setup(this, defStream);

	HaloExchanger halo(cartComm);
	Redistributor redist(cartComm);

	for (int i=0; i<particleVectors.size(); i++)
	{
		auto it = std::max_element(cellListTable[i].begin(), cellListTable[i].end(),
				[] (CellList* cl1, CellList* cl2) { return cl1->rc < cl2->rc; } );
		halo.attach(particleVectors[i], *it);
		redist.attach(particleVectors[i], *it);
	}

	float t = 0;
	for (int iter=0; i<nsteps; i++)
	{
		//===================================================================================================
		for (auto& pv : particleVectors)
			pv->forces.clear(defStream);

		//===================================================================================================
		for (auto& cllist : cellListTable)
			for (auto& cl : cllist)
				cl->build(defStream);

		for (auto& p : plugins)
			p->beforeForces(t);

		//===================================================================================================
		for (int i=0; i<interactionTable.size(); i++)
			for (int j=0; j<interactionTable[i].size(); j++)
				if (interactionTable[i][j].first != nullptr)
				{
					if (i == j)
						interactionTable[i][j].first.execSelf(particleVectors[i], interactionTable[i][j].second, t, defStream);
					else
						interactionTable[i][j].first.execExternal(particleVectors[i], particleVectors[j], interactionTable[i][j].second, t, defStream);
				}

		//===================================================================================================
		halo.exchange();

		//===================================================================================================
		for (int i=0; i<interactionTable.size(); i++)
			for (int j=0; j<interactionTable[i].size(); j++)
				if (interactionTable[i][j].first != nullptr)
				{
					interactionTable[i][j].first.execHalo(particleVectors[i], interactionTable[i][j].second, t, defStream);
				}

		for (auto& p : plugins)
			p->beforeIntegration(t);

		//===================================================================================================
		for (int i=0; i<integrators.size(); i++)
			integrators[i]->exec(particleVectors[i], dt, defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );

		for (auto& p : plugins)
			p->afterIntegration(t);

		//===================================================================================================
		redist.redistribute();
		CUDA_Check( cudaStreamSynchronize(defStream) );
	}
}


//===================================================================================================
// Postprocessing
//===================================================================================================

Postprocess::Postprocess(MPI_Comm& comm) : comm(comm) {};

void Postprocess::registerPlugin(PostprocessPlugin* plugin)
{
	plugins.push_back(plugin);
}



//===================================================================================================
// uDeviceX
//===================================================================================================

void uDeviceX::run()
{
	int nranks, rank;

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

	MPI_Comm  iocomm, activecomm, intercomm, splitcomm;

	if (nranks % 2 != 0)
		die("Number of MPI ranks should be even");

	int computeTask = (rank+1) % 2;
	MPI_Check( MPI_Comm_split(MPI_COMM_WORLD, computeTask, rank, &splitcomm) );

	if (computeTask)
	{
		MPI_Check( MPI_Comm_dup(splitcomm, &activecomm) );
		MPI_Check( MPI_Intercomm_create(activecomm, 0, MPI_COMM_WORLD, 1, 0, &intercomm) );
	}
	else
	{
		MPI_Check( MPI_Comm_dup(splitcomm, &iocomm) );
		MPI_Check( MPI_Intercomm_create(iocomm,     0, MPI_COMM_WORLD, 0, 0, &intercomm) );
	}

	if (computeTask)
	{
		MPI_Check( MPI_Barrier(activecomm) );
		sim->run(nsteps);

		MPI_Check( MPI_Comm_free(&activecomm) );
		MPI_Check( MPI_Comm_free(&intercomm) );
	}
	else
	{
		MPI_Check( MPI_Barrier(iocomm) );
		post->run();

		MPI_Check( MPI_Comm_free(&iocomm) );
		MPI_Check( MPI_Comm_free(&intercomm) );
	}


	MPI_Check( MPI_Finalize() );

	if (computeTask)
	{
		CUDA_Check( cudaDeviceSynchronize() );
		CUDA_Check(cudaDeviceReset());
	}
}




