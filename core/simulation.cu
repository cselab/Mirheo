#include <algorithm>

#include <core/simulation.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/redistributor.h>
#include <core/halo_exchanger.h>
#include <core/logger.h>

Simulation::Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm) :
nranks3D(nranks3D), globalDomainSize(globalDomainSize), interComm(interComm), currentTime(0), currentStep(0)
{
	int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
	int periods[] = {1, 1, 1};
	int coords[3];

	MPI_Check( MPI_Comm_rank(comm, &rank) );
	MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
	MPI_Check( MPI_Cart_get(cartComm, 3, ranksArr, periods, coords) );
	rank3D = {coords[0], coords[1], coords[2]};

	subDomainSize = globalDomainSize / make_float3(nranks3D);
	subDomainStart = {subDomainSize.x * coords[0], subDomainSize.y * coords[1], subDomainSize.z * coords[2]};

	debug("Simulation initialized");
}

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;
	particleVectors.push_back(pv);

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;
	ic->exec(cartComm, pv, globalDomainSize, subDomainSize);
}

void Simulation::registerObjectVector  (ObjectVector* ov)
{
	std::string name = ov->name;
	particleVectors.push_back(static_cast<ParticleVector*>(ov));

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;
}

void Simulation::registerWall          (Wall* wall)
{
	std::string name = wall->name;

	if (wallMap.find(name) != wallMap.end())
		die("More than one wall is called %s", name.c_str());

	if (pvIdMap.find(name) != pvIdMap.end())
		die("Wall has the same name as particle vector: %s", name.c_str());

	wallMap[name] = wall;

	particleVectors.push_back(wall->getFrozen());
	pvIdMap[name] = particleVectors.size() - 1;
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
	if (pvIdMap.find(pvName) == pvIdMap.end())
		die("No such particle vector: %s", pvName.c_str());

	if (integratorMap.find(integratorName) == integratorMap.end())
		die("No such integrator: %s", integratorName.c_str());

	const int pvId = pvIdMap[pvName];
	integrators.resize(std::max((int)integrators.size(), pvId+1), nullptr);
	integrators[pvId] = integratorMap[integratorName];
}

void Simulation::setInteraction(std::string pv1Name, std::string pv2Name, std::string interactionName)
{
	if (pvIdMap.find(pv1Name) == pvIdMap.end())
		die("No such particle vector: %s", pv1Name.c_str());

	if (pvIdMap.find(pv2Name) == pvIdMap.end())
		die("No such particle vector: %s", pv2Name.c_str());

	if (interactionMap.find(interactionName) == interactionMap.end())
		die("No such integrator: %s", interactionName.c_str());

	auto pv1Id = pvIdMap[pv1Name];
	auto pv2Id = pvIdMap[pv2Name];
	auto pv1 = particleVectors[pv1Id];
	auto pv2 = particleVectors[pv2Id];
	auto interaction = interactionMap[interactionName];
	float rc = interaction->rc;

	CellList *cl1, *cl2;
	cellListMaps.resize( std::max({pv1Id+1, pv2Id+1, (int)cellListMaps.size()}) );
	if (cellListMaps[pv1Id].find(rc) != cellListMaps[pv1Id].end())
		cl1 = cellListMaps[pv1Id][rc];
	else
		cellListMaps[pv1Id][rc] = cl1 = new CellList(particleVectors[pv1Id], rc, subDomainSize);

	if (cellListMaps[pv2Id].find(rc) != cellListMaps[pv2Id].end())
		cl2 = cellListMaps[pv2Id][rc];
	else
		cellListMaps[pv2Id][rc] = cl2 = new CellList(particleVectors[pv2Id], rc, subDomainSize);

	auto frc = [=] (InteractionType type, float t, cudaStream_t stream) {
		cl1->build(stream, true);
		cl2->build(stream, true);
		interaction->exec(type, pv1, pv2, cl1, cl2, t, stream);
	};

	forceCallers.insert({rc, frc});
}

std::vector<int> Simulation::getWallCreationSteps() const
{
	std::vector<int> res;
	for (auto wall : wallMap)
		res.push_back(wall.second->creationTime());

	std::sort(res.begin(), res.end());
	return res;
}

void Simulation::init()
{
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 0) );

	debug("Simulation initiated, preparing plugins");
	for (auto& pl : plugins)
	{
		pl->setup(this, defStream, cartComm, interComm);
		pl->handshake();
	}

	halo = new HaloExchanger(cartComm, defStream);
	redistributor = new Redistributor(cartComm);

	debug("Attaching particle vectors to halo exchanger and redistributor");
	for (int i=0; i<particleVectors.size(); i++)
		if (cellListMaps[i].size() > 0 && particleVectors[i]->np > 0)
		{
			auto cl = cellListMaps[i].begin()->second;

			halo->attach         (particleVectors[i], cl);
			redistributor->attach(particleVectors[i], cl);
		}

	// Manage streams
	// XXX: Is it really needed?
	debug("Setting up streams");
	for (auto pv : particleVectors)
		pv->pushStreamWOhalo(defStream);

	for (auto clMap : cellListMaps)
		for (auto rc_cl : clMap)
			rc_cl.second->setStream(defStream);
}

void Simulation::createWalls()
{
	for (auto wall : wallMap)
	{
		wall.second->create(cartComm, subDomainStart, subDomainSize, globalDomainSize, particleVectors[pvIdMap["dpd"]]);
		wall.second->attach(particleVectors[pvIdMap["dpd"]], cellListMaps[pvIdMap["dpd"]].begin()->second);
		halo->attach( wall.second->getFrozen(), cellListMaps[pvIdMap[wall.first]].begin()->second );
	}
}

// TODO: wall has self-interactions
void Simulation::run(int nsteps)
{
	// XXX: different dt not implemented yet
	float dt = 1e9;
	for (auto integr : integrators)
		if (integr != nullptr)
			dt = min(dt, integr->dt);

	info("Will run %d iterations now", nsteps);
	int begin = currentStep, end = currentStep + nsteps;
	for (currentStep = begin; currentStep < end; currentStep++)
	{
		if (rank == 0)
			info("===============================================================================\nTimestep: %d, simulation time: %f",
					currentStep, currentTime);
		//===================================================================================================

		debug("Building halo cell-lists");
		for (auto clMap : cellListMaps)
			if (clMap.size() > 0)
				clMap.begin()->second->build(defStream);

		//===================================================================================================

		for (auto& pv : particleVectors)
			pv->forces.clear();

		debug("Plugins: before forces");
		for (auto& pl : plugins)
		{
			pl->setTime(currentTime, currentStep);
			pl->beforeForces();
		}

		//===================================================================================================

		// XXX: Not overlapped (do we need overlap at all?)
		debug("Initializing halo exchange");
		halo->init();

		//===================================================================================================

		// TODO: Forces need to be rearranged here as well
		debug("Computing internal forces");
		for (auto& rc_frc : forceCallers)
			rc_frc.second(InteractionType::Regular, currentTime, defStream);

		//===================================================================================================

		debug("Plugins: serialize and send");
		for (auto& pl : plugins)
			pl->serializeAndSend();

		//===================================================================================================

		debug("Finalizing halo exchange");
		halo->finalize();

		debug("Computing halo forces");
		// Iterate over cell-lists in reverse order, because the last CLs are already built
		for (auto rc_frc = forceCallers.rbegin(); rc_frc != forceCallers.rend(); rc_frc++)
			rc_frc->second(InteractionType::Halo, currentTime, defStream);

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

		// TODO: correct dt should be attached to the wall
		debug("Bounce from the walls");
		for (auto wall : wallMap)
			wall.second->bounce(dt, defStream);

		//===================================================================================================

		// XXX: probably should go after redistr AND after cell-list
		debug("Plugins: after integration");
		for (auto pl : plugins)
			pl->afterIntegration();

		//===================================================================================================

		debug("Redistributing particles, cell-lists may need to be updated");
		for (auto clMap : cellListMaps)
			if (clMap.size() > 0)
				clMap.begin()->second->build(defStream);

		CUDA_Check( cudaStreamSynchronize(defStream) );
		redistributor->redistribute();

		//===================================================================================================

		currentTime += dt;
	}

	info("Finished with %d iterations", nsteps);
}

void Simulation::finalize()
{
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

void uDeviceX::run(int nsteps)
{
	if (isComputeTask())
	{
		auto wallCreationSteps = sim->getWallCreationSteps();

		sim->init();

		int totSteps = 0;
		for (int i=0; i<wallCreationSteps.size(); i++)
		{
			sim->run(wallCreationSteps[i] - totSteps);
			sim->createWalls();
			totSteps += wallCreationSteps[i];
		}
		sim->run(nsteps - totSteps);

		sim->finalize();

		CUDA_Check( cudaDeviceSynchronize() );
	}
	else
		post->run();
}




