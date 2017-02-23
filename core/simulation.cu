#include <algorithm>

#include <core/simulation.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/redistributor.h>
#include <core/halo_exchanger.h>
#include <core/logger.h>

Simulation::Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm) :
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

	debug("Simulation initialized");
}

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;
	particleVectors.push_back(pv);

	if (pvMap.find(name) != pvMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvMap[name] = particleVectors.size() - 1;
	ic->exec(cartComm, pv, globalDomainSize, subDomainSize);
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
	auto interaction = interactionMap[interactionName];

	// Allocate interactionTable
	interactionTable.resize(std::max((int)interactionTable.size(), pv1Id+1));
	auto& interactionsMap = interactionTable[pv1Id];

	std::vector<std::pair<Interaction*, int>>* interactionVector = nullptr;
	for (auto& entry : interactionsMap)
	{
		if (fabs(entry.first->rc - interaction->rc) < 1e-6)
		{
			interactionVector = &entry.second;
			break;
		}
	}

	if (interactionVector == nullptr)
	{
		auto cl = new CellList(particleVectors[pv1Id], interaction->rc, -subDomainSize*0.5, subDomainSize);
		interactionVector = new std::vector<std::pair<Interaction*, int>>;
		interactionVector->push_back( {interaction, pv2Id} );

		interactionTable[pv1Id][cl] = *interactionVector;
	}
	else
	{
		interactionVector->push_back( {interaction, pv2Id} );
	}

	largestRC.resize( std::max({pv1Id+1, pv2Id+1, (int)largestRC.size()}), -1.0f );

	largestRC[pv1Id] = std::max(largestRC[pv1Id], interaction->rc);
	largestRC[pv2Id] = std::max(largestRC[pv2Id], interaction->rc);
}

// TODO: wall has self-interactions
void Simulation::run(int nsteps)
{
	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	debug("Simulation initiated, preparing plugins");
	for (auto& pl : plugins)
	{
		pl->setup(this, defStream, cartComm, interComm);
		pl->handshake();
	}

	HaloExchanger halo(cartComm);
	Redistributor redist(cartComm);

	debug("Determining halo and redistributor cell-lists for each PV");
	std::vector<CellList*> haloLists  (particleVectors.size(), nullptr);
	std::vector<CellList*> redistLists(particleVectors.size(), nullptr);
	interactionTable.resize(particleVectors.size());
	largestRC.resize(particleVectors.size(), -1);

	for (int i=0; i<particleVectors.size(); i++)
	{
		// We need the biggest cell-list for the halo
		if (interactionTable[i].size() == 0 || largestRC[i] > interactionTable[i].begin()->first->rc)
		{
			if (largestRC[i] > 0.0f)
				haloLists[i] = new CellList(particleVectors[i], largestRC[i], -subDomainSize*0.5, subDomainSize);
		}
		else
			haloLists[i] = interactionTable[i].begin()->first;

		// Redistribution is ok with ANY cell-list
		// so we choose the first cell-list in the interaction table
		// if it exists, or the main one
		if (interactionTable[i].size() == 0)
			redistLists[i] = haloLists[i];
		else
			redistLists[i] = interactionTable[i].begin()->first;
	}

	debug("Attaching particle vectors to halo exchanger and redistributor");
	for (int i=0; i<particleVectors.size(); i++)
	{
		if (haloLists[i] != nullptr)
			halo.attach  (particleVectors[i], haloLists[i]);

		if (redistLists[i] != nullptr)
			redist.attach(particleVectors[i], redistLists[i]);

		// Manage streams
		// XXX: Is it really needed?
		particleVectors[i]->pushStreamWOhalo(defStream);
		for (auto& cl_vec : interactionTable[i])
			cl_vec.first->setStream(defStream);
	}

	// XXX: different dt not implemented yet
	float dt = 1e9;
	for (auto integr : integrators)
		if (integr != nullptr)
			dt = min(dt, integr->dt);

	float t = 0;
	debug("Started simulation");
	for (int iter=0; iter<nsteps; iter++)
	{
		if (rank == 0)
			info("===============================================================================\nTimestep: %d, simulation time: %f", iter, t);
		//===================================================================================================

		debug("Building halo cell-lists");
		for (auto cl : haloLists)
			if (cl != nullptr) cl->build(defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );

		//===================================================================================================

		for (auto& pv : particleVectors)
			pv->forces.clear();

		debug("Plugins: before forces");
		for (auto& pl : plugins)
		{
			pl->setTime(t, iter);
			pl->beforeForces();
		}

		//===================================================================================================


		// We can compute now the forces ONLY for the halo cell-lists because they are ready
		debug("Computing forces with halo cell-lists");
		for (int i=0; i<interactionTable.size(); i++)
		{
			for (auto& cl_list : interactionTable[i])
			{
				auto& cl = cl_list.first;
				auto& intList = cl_list.second;

				if (cl == haloLists[i])
					for (auto& entry : intList)
					{
						auto& interactionExec = entry.first;
						int j = entry.second;

						if (i == j)
							interactionExec->execSelf(particleVectors[i], cl, t, defStream);
						else
							interactionExec->execExternal(particleVectors[i], particleVectors[j], cl, t, defStream);
					}
			}
		}

		// Overlapped with previous forces
		debug("Initializing halo exchange");
		halo.init();

		// Now continue with the forces for all the rest cell-lists
		debug("Computing forces with all the other cell-lists");
		for (int i=0; i<interactionTable.size(); i++)
		{
			for (auto& cl_list : interactionTable[i])
			{
				auto& cl = cl_list.first;
				auto& intList = cl_list.second;

				if (cl != haloLists[i])
				{
					cl->build(defStream);
					for (auto& entry : intList)
					{
						auto& interactionExec = entry.first;
						int j = entry.second;

						if (i == j)
							interactionExec->execSelf(particleVectors[i], cl, t, defStream);
						else
							interactionExec->execExternal(particleVectors[i], particleVectors[j], cl, t, defStream);
					}
				}
			}
		}

		//===================================================================================================

		debug("Plugins: serialize and send");
		for (auto& pl : plugins)
			pl->serializeAndSend();

		//===================================================================================================

		debug("Finalizing halo exchange");
		halo.finalize();

		debug("Computing halo forces");
		// Iterate over cell-lists in reverse order, because the last CL is already build
		for (int i=0; i<interactionTable.size(); i++)
		{
			for (auto cl_listIter = interactionTable[i].rbegin(); cl_listIter != interactionTable[i].rend(); cl_listIter++)
			{
				auto& cl = cl_listIter->first;
				auto& intList = cl_listIter->second;

				if (cl_listIter != interactionTable[i].rbegin())
					cl->build(defStream);

				for (auto& entry : intList)
				{
					auto& interactionExec = entry.first;
					int j = entry.second;

					interactionExec->execHalo(particleVectors[i], particleVectors[j], cl, t, defStream);
				}
			}
		}

		//===================================================================================================

		debug("Plugins: before integration");
		for (auto& pl : plugins)
			pl->beforeIntegration();

		//===================================================================================================

		debug("Performing integration");
		for (int i=0; i<integrators.size(); i++)
			if (integrators[i] != nullptr)
				integrators[i]->exec(particleVectors[i], defStream);

		//===================================================================================================

		// XXX: probably should go after redistr AND after cell-list
		debug("Plugins: after integration");
		bool reordered = false;
		for (auto& pl : plugins)
		{
			bool oneReord = false;
			pl->afterIntegration(oneReord);
			reordered = reordered || oneReord;
		}

		//===================================================================================================

		debug("Redistributing particles");
		if (reordered)
		{
			debug("Some plugins have changed ordering of the particles, rebuilding cell-lists for the redistribution");
			for (auto cl : haloLists)
				if (cl != nullptr) cl->build(defStream);
		}
		CUDA_Check( cudaStreamSynchronize(defStream) );
		redist.redistribute();


		t += dt;
	}

	MPI_Check( MPI_Barrier(cartComm) );

	debug("Finished, exiting now");

	if (interComm != MPI_COMM_NULL)
	{
		int dummy = -1;
		int tag = 424242;

		MPI_Request req;
		MPI_Check( MPI_Isend(&dummy, 1, MPI_INT, rank, tag, interComm, &req) );
	}
}


//===================================================================================================
// Postprocessing
//===================================================================================================

Postprocess::Postprocess(MPI_Comm& comm, MPI_Comm& interComm) : comm(comm), interComm(interComm)
{
	debug("Postprocessing initialized");
}

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

	// Stopping condition
	int dummy = 0;
	int tag = 424242;
	int rank;

	MPI_Check( MPI_Comm_rank(comm, &rank) );

	MPI_Request endReq;
	MPI_Check( MPI_Irecv(&dummy, 1, MPI_INT, rank, tag, interComm, &endReq) );

	std::vector<MPI_Request> requests;
	for (auto& pl : plugins)
		requests.push_back(pl->postRecv());
	requests.push_back(endReq);

	while (true)
	{
		int index;
		MPI_Status stat;
		MPI_Check( MPI_Waitany(requests.size(), requests.data(), &index, &stat) );

		if (index == plugins.size())
		{
			if (dummy != -1)
				die("Something went terribly wrong");

			// TODO: Maybe cancel?
			debug("Postprocess got a stopping message and will exit now");
			break;
		}

		debug2("Postprocess got a request from plugin %s, executing now", plugins[index]->name.c_str());
		plugins[index]->deserialize(stat);
		requests[index] = plugins[index]->postRecv();
	}
}



//===================================================================================================
// uDeviceX
//===================================================================================================

uDeviceX::uDeviceX(int argc, char** argv, int3 nranks3D, float3 globalDomainSize,
		Logger& logger, std::string logFileName, int verbosity, bool noPostprocess) : noPostprocess(noPostprocess)
{
	int nranks, rank;

//	int provided;
//	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
//	if (provided < MPI_THREAD_MULTIPLE)
//	{
//		printf("ERROR: The MPI library does not have full thread support\n");
//		MPI_Abort(MPI_COMM_WORLD, 1);
//	}

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

	info("Program started, splitting commuticator");

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

void uDeviceX::run()
{
	if (isComputeTask())
	{
		sim->run(16000);
	}
	else
		post->run();

	if (computeTask)
	{
		CUDA_Check( cudaDeviceSynchronize() );
		CUDA_Check( cudaDeviceReset() );
	}

	MPI_Check( MPI_Finalize() );
}




