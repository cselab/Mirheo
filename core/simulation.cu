#include "simulation.h"

#include <algorithm>

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

	localDomainSize = globalDomainSize / make_float3(nranks3D);
	globalDomainStart = {localDomainSize.x * coords[0], localDomainSize.y * coords[1], localDomainSize.z * coords[2]};

	restartFolder  = "./restart/";
	std::string command = "mkdir -p " + restartFolder;
	if (rank == 0)
	{
		if ( system(command.c_str()) != 0 )
		{
			error("Could not create folder for restart files, will try to use ./");
			restartFolder = "./";
		}
	}

	debug("Simulation initialized, subdomain size is [%f %f %f], subdomain starts at [%f %f %f]",
			localDomainSize.x,  localDomainSize.y,  localDomainSize.z,
			globalDomainStart.x, globalDomainStart.y, globalDomainStart.z);
}

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;
	particleVectors.push_back(pv);

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;
	ic->exec(cartComm, pv, globalDomainStart, localDomainSize, 0);
}

void Simulation::registerObjectVector(ObjectVector* ov)
{
	std::string name = ov->name;
	particleVectors.push_back(static_cast<ParticleVector*>(ov));

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;
}

void Simulation::registerWall(Wall* wall, bool addCorrespondingPV)
{
	std::string name = wall->name;

	if (wallMap.find(name) != wallMap.end())
		die("More than one wall is called %s", name.c_str());

	wallMap[name] = wall;
	wall->createSdf(cartComm, globalDomainSize, globalDomainStart, localDomainSize);

	if (addCorrespondingPV)
	{
		if (pvIdMap.find(name) != pvIdMap.end())
			die("Wall has the same name as particle vector: %s", name.c_str());

		auto pv = new ParticleVector(wall->name);
		pv->globalDomainStart = globalDomainStart;
		pv->localDomainSize = localDomainSize;
		pv->restart(cartComm, restartFolder);
		particleVectors.push_back(pv);
		pvIdMap[name] = particleVectors.size() - 1;
	}
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

	interactionPrototypes.push_back(std::make_tuple(rc, pv1, pv2, interaction));
}

void Simulation::init()
{
	const float rcTolerance = 1e-4;

	std::map<ParticleVector*, std::vector<float>> cutOffMap;

	// Deal with the cell-lists and interactions
	for (auto prototype : interactionPrototypes)
	{
		float rc = std::get<0>(prototype);
		cutOffMap[std::get<1>(prototype)].push_back(rc);
		cutOffMap[std::get<2>(prototype)].push_back(rc);
	}

	for (auto& cutoffs : cutOffMap)
	{
		std::sort(cutoffs.second.begin(), cutoffs.second.end(), [] (float a, float b) { return a > b; });

		auto it = std::unique(cutoffs.second.begin(), cutoffs.second.end(), [=] (float a, float b) { return fabs(a - b) < rcTolerance; });
		cutoffs.second.resize( std::distance(cutoffs.second.begin(), it) );

		bool first = true;
		for (auto rc : cutoffs.second)
		{
			cellListMap[cutoffs.first].push_back(first ? new PrimaryCellList(cutoffs.first, rc, localDomainSize) : new CellList(cutoffs.first, rc, localDomainSize));
			first = false;
		}
	}

	for (auto prototype : interactionPrototypes)
	{
		float rc = std::get<0>(prototype);
		auto pv1 = std::get<1>(prototype);
		auto pv2 = std::get<2>(prototype);

		auto& clVec1 = cellListMap[pv1];
		auto& clVec2 = cellListMap[pv2];

		CellList *cl1, *cl2;

		for (auto cl : clVec1)
			if (fabs(cl->rc - rc) <= rcTolerance)
				cl1 = cl;

		for (auto cl : clVec2)
			if (fabs(cl->rc - rc) <= rcTolerance)
				cl2 = cl;

		auto inter = std::get<3>(prototype);

		regularInteractions.push_back([inter, pv1, pv2, cl1, cl2] (float t, cudaStream_t stream) {
			inter->regular(pv1, pv2, cl1, cl2, t, stream);
		});

		haloInteractions.push_back([inter, pv1, pv2, cl1, cl2] (float t, cudaStream_t stream) {
			inter->halo(pv1, pv2, cl1, cl2, t, stream);
		});
	}

	debug("Simulation initiated, preparing plugins");
	for (auto& pl : plugins)
	{
		pl->setup(this, cartComm, interComm);
		pl->handshake();
	}

	halo = new ParticleHaloExchanger(cartComm);
	redistributor = new ParticleRedistributor(cartComm);

	debug("Attaching particle vectors to halo exchanger and redistributor");
	for (auto pv : particleVectors)
		if (cellListMap[pv].size() > 0)
		{
			auto cl = cellListMap[pv][0];

			halo->attach         (pv, cl);
			redistributor->attach(pv, cl);
		}

	// Remove stuff from inside the wall, attach for bounce
	for (auto pv : particleVectors)
		for (auto wall : wallMap)
			if (cellListMap[pv].size() > 0 && pv->name != wall.second->name)
			{
				wall.second->removeInner(pv);

				// TODO: select what to bounce
				wall.second->attach(pv, cellListMap[pv][0]);
			}

	assemble();
}

void Simulation::assemble()
{
	// XXX: different dt not implemented
	dt = 1.0;
	for (auto integr : integrators)
		if (integr != nullptr)
			dt = min(dt, integr->dt);


	scheduler.addTask("Сell-lists", [&] (cudaStream_t stream) {
		for (auto clVec : cellListMap)
			for (auto cl : clVec.second)
				cl->build(stream);
	});

	scheduler.addTask("Clear forces", [&] (cudaStream_t stream) {
		for (auto& pv : particleVectors)
			pv->local()->forces.clear(stream);
	});

	scheduler.addTask("Plugins: before forces", [&] (cudaStream_t stream) {
		for (auto& pl : plugins)
			{
				pl->setTime(currentTime, currentStep);
				pl->beforeForces(stream);
			}
	});

	scheduler.addTask("Halo init", [&] (cudaStream_t stream) {
		halo->init(stream);
	});

	scheduler.addTask("Internal forces", [&] (cudaStream_t stream) {
		for (auto& inter : regularInteractions)
			inter(currentTime, stream);
	});

	scheduler.addTask("Plugins: serialize and send", [&] (cudaStream_t stream) {
		for (auto& pl : plugins)
			pl->serializeAndSend(stream);
	});

	scheduler.addTask("Halo finalize", [&] (cudaStream_t stream) {
		halo->finalize();
	});

	scheduler.addTask("Halo forces", [&] (cudaStream_t stream) {
		for (auto& inter : haloInteractions)
			inter(currentTime, stream);
	});

	scheduler.addTask("Accumulate forces", [&] (cudaStream_t stream) {
		for (auto clVec : cellListMap)
			for (auto cl : clVec.second)
				cl->addForces(stream);
	});

	scheduler.addTask("Plugins: before integration", [&] (cudaStream_t stream) {
		for (auto& pl : plugins)
			pl->beforeIntegration(stream);
	});

	scheduler.addTask("Integration", [&] (cudaStream_t stream) {
		for (int i=0; i<integrators.size(); i++)
			if (integrators[i] != nullptr)
				integrators[i]->stage2(particleVectors[i], stream);
	});

	scheduler.addTask("Wall bounce", [&] (cudaStream_t stream) {
		for (auto wall : wallMap)
		{
			wall.second->bounce(dt, stream);
			wall.second->check(stream);
		}
	});

	scheduler.addTask("Plugins: after integration", [&] (cudaStream_t stream) {
		for (auto pl : plugins)
			pl->afterIntegration(stream);
	});

	scheduler.addTask("Redistribute init", [&] (cudaStream_t stream) {
		redistributor->init(stream);
	});

	scheduler.addTask("Redistribute finalize", [&] (cudaStream_t stream) {
		redistributor->finalize();
	});

	// cell lists
	// plugins before forces
	// init part halo
	// internal object forces
	// init obj halo
	// internal particle-only forces
	// both halos finalize
	// halo forces
	// send back obj forces  --> possible to move up and overlap
	// plugins before integration
	// integrate all internal
	// integrate EXTERNAL objects
	// plugins after integration
	// wall bounce
	// object bounce with EXTERNAL included
	// send back obj forces
	// redistribute


	scheduler.addDependency("Сell-lists", {"Internal forces", "Halo init"}, {});
	scheduler.addDependency("Plugins: before forces", {"Internal forces", "Halo init"}, {});
	scheduler.addDependency("Internal forces", {}, {"Clear forces"});
	scheduler.addDependency("Plugins: serialize and send", {}, {"Internal forces"});
	scheduler.addDependency("Halo init", {"Internal forces"}, {});
	scheduler.addDependency("Halo finalize", {}, {"Halo init"});
	scheduler.addDependency("Halo forces", {}, {"Halo finalize"});
	scheduler.addDependency("Accumulate forces", {"Integration"}, {"Halo forces", "Internal forces"});
	scheduler.addDependency("Plugins: before integration", {"Integration"}, {});
	scheduler.addDependency("Wall bounce", {}, {"Integration"});
	scheduler.addDependency("Plugins: after integration", {}, {"Integration", "Wall bounce"});
	scheduler.addDependency("Redistribute init", {}, {"Integration", "Wall bounce", "Plugins: after integration"});
	scheduler.addDependency("Redistribute finalize", {}, {"Redistribute init"});

	scheduler.compile();

}

// TODO: wall has self-interactions
void Simulation::run(int nsteps)
{
	info("Will run %d iterations now", nsteps);
	int begin = currentStep, end = currentStep + nsteps;

	for (currentStep = begin; currentStep < end; currentStep++)
	{
		if (rank == 0)
			info("===============================================================================\nTimestep: %d, simulation time: %f",
					currentStep, currentTime);

		scheduler.run();

		currentTime += dt;
	}

	// Finish the redistribution by rebuilding the primary cell-lists
	for (auto clVec : cellListMap)
		if (clVec.second.size() > 0)
			clVec.second[0]->build(0);

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
}




