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

	info("Simulation initialized, subdomain size is [%f %f %f], subdomain starts at [%f %f %f]",
			localDomainSize.x,  localDomainSize.y,  localDomainSize.z,
			globalDomainStart.x, globalDomainStart.y, globalDomainStart.z);
}

//================================================================================================
// Registration
//================================================================================================

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;
	particleVectors.push_back(pv);

	auto ov = dynamic_cast<ObjectVector*>(pv);
	if(ov != nullptr)
		objectVectors.push_back(ov);

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	if (wallMap.find(name) != wallMap.end())
		die("Particle vector cannot be called as another wall %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;
	ic->exec(cartComm, pv, globalDomainStart, localDomainSize, 0);
}

void Simulation::registerWall(Wall* wall)
{
	std::string name = wall->name;

	if (wallMap.find(name) != wallMap.end())
		die("More than one wall is called %s", name.c_str());

	if (pvIdMap.find(name) != pvIdMap.end())
		die("Wall cannot be called as another particle vector %s", name.c_str());

	wallMap[name] = wall;
	wall->createSdf(cartComm, globalDomainSize, globalDomainStart, localDomainSize);
}

void Simulation::registerInteraction(Interaction* interaction)
{
	std::string name = interaction->name;
	if (interactionMap.find(name) != interactionMap.end())
		die("More than one interaction is called %s", name.c_str());

	interactionMap[name] = interaction;
}

void Simulation::registerIntegrator(Integrator* integrator)
{
	std::string name = integrator->name;
	if (integratorMap.find(name) != integratorMap.end())
		die("More than one interaction is called %s", name.c_str());

	integratorMap[name] = integrator;
}

void Simulation::registerBouncer(Bouncer* bouncer)
{
	std::string name = bouncer->name;
	if (bouncerMap.find(name) != bouncerMap.end())
		die("More than one bouncer is called %s", name.c_str());

	bouncerMap[name] = bouncer;
}

void Simulation::registerPlugin(SimulationPlugin* plugin)
{
	plugins.push_back(plugin);
}

//================================================================================================
// Applying something to something else
//================================================================================================

void Simulation::setIntegrator(std::string integratorName, std::string pvName)
{
	if (integratorMap.find(integratorName) == integratorMap.end())
		die("No such integrator: %s", integratorName.c_str());

	auto pv = getPVbyName(pvName);
	if (pv == nullptr)
		die("No such particle vector: %s", pvName.c_str());

	integrator = integratorMap[integratorName];

	integratorsStage1.push_back([integrator, pv] (cudaStream_t stream) {
		integrator->stage1(pv, stream);
	});

	integratorsStage2.push_back([integrator, pv] (cudaStream_t stream) {
		integrator->stage1(pv, stream);
	});
}

void Simulation::setInteraction(std::string interactionName, std::string pv1Name, std::string pv2Name)
{
	auto pv1 = getPVbyName(pv1Name);
	if (pv1 == nullptr)
		die("No such particle vector: %s", pv1Name.c_str());

	auto pv2 = getPVbyName(pv2Name);
	if (pv2 == nullptr)
		die("No such particle vector: %s", pv2Name.c_str());

	if (interactionMap.find(interactionName) == interactionMap.end())
		die("No such integrator: %s", interactionName.c_str());
	auto interaction = interactionMap[interactionName];


	float rc = interaction->rc;
	interactionPrototypes.push_back(std::make_tuple(rc, pv1, pv2, interaction));
}

void Simulation::setBouncer(std::string bouncerName, std::string objName, std::string pvName)
{
	auto pv = getPVbyName(pvName);
	if (pv == nullptr)
		die("No such particle vector: %s", pvName.c_str());

	auto ov = dynamic_cast<ObjectVector*> (getPVbyName(objName));
	if (ov == nullptr)
		die("No such object vector: %s", objName.c_str());

	if (bouncerMap.find(bouncerName) == bouncerMap.end())
		die("No such bouncer: %s", bouncerName.c_str());
	auto bouncer = bouncerMap[bouncerName];

	bouncerPrototypes.push_back(std::make_tuple(bouncer, ov, pv));
}

void Simulation::setWallBounce(std::string wallName, std::string pvName)
{
	auto pv = getPVbyName(pvName);
	if (pv == nullptr)
		die("No such particle vector: %s", pvName.c_str());

	if (wallMap.find(wallName) == wallMap.end())
		die("No such wall: %s", wallName.c_str());
	auto wall = wallMap[wallName];

	wallPrototypes.push_back( {wall, pv} );
}


void Simulation::prepareCellLists()
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

		bool primary = true;

		// Don't use primary cell-lists with ObjectVectors
		if (dynamic_cast<ObjectVector*>(cutoffs.first) != nullptr)
			primary = false;

		for (auto rc : cutoffs.second)
		{
			cellListMap[cutoffs.first].push_back(first ?
					new PrimaryCellList(cutoffs.first, rc, localDomainSize) :
					new CellList       (cutoffs.first, rc, localDomainSize));
			primary = false;
		}
	}
}

void Simulation::prepareInteractions()
{
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
}

void Simulation::prepareBouncers()
{
	for (auto prototype : bouncerPrototypes)
	{
		auto bouncer = std::get<0>(prototype);
		auto ov = std::get<1>(prototype);
		auto pv = std::get<2>(prototype);

		auto& clVec = cellListMap[pv];

		if (clVec.empty()) continue;

		CellList *cl = clVec[0];

		regularBouncers.push_back([bouncer, ov, pv, cl] (cudaStream_t stream) {
			bouncer->bounceLocal(ov, pv, cl, stream);
		});

		haloBouncers.   push_back([bouncer, ov, pv, cl] (cudaStream_t stream) {
			bouncer->bounceLocal(ov, pv, cl, stream);
		});
	}
}

void Simulation::prepareWalls()
{
	for (auto prototype : wallPrototypes)
	{
		auto wall = prototype.first;
		auto pv   = prototype.second;

		auto& clVec = cellListMap[pv];

		if (clVec.empty()) continue;

		CellList *cl = clVec[0];

		wall->attach(pv, cl);
	}

	for (auto pv : particleVectors)
		for (auto wall : wallMap)
			if (cellListMap[pv].size() > 0 && pv->name != wall.second->name)
				wall.second->removeInner(pv);
}

void Simulation::init()
{
	prepareCellLists();

	prepareInteractions();
	prepareBouncers();
	prepareWalls();

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
			if (dynamic_cast<ObjectVector*>(pv) == nullptr)
			{
				auto cl = cellListMap[pv][0];

				halo->attach         (pv, cl);
				redistributor->attach(pv, cl);
			}
			else
			{
				auto cl = cellListMap[pv][0];
				auto ov = dynamic_cast<ObjectVector*>(pv);

				objHalo->attach        (ov, cl->rc);
				objRedistibutor->attach(ov, cl->rc);
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


	scheduler.addTask("Object halo init", [&] (cudaStream_t stream) {
		objHalo->init(stream);
	});
	scheduler.addTask("Object halo finalize", [&] (cudaStream_t stream) {
		objHalo->finalize();
	});

	scheduler.addTask("Object bounce", [&] (cudaStream_t stream) {
		for (auto bouncer : bouncers)
			bouncer.first->exec(dt, bouncer.second, stream);
	});

	scheduler.addTask("Obj forces exchange: init", [&] (cudaStream_t stream) {
		objForceExchanger->init(stream);
	});

	scheduler.addTask("Obj forces exchange: finalize", [&] (cudaStream_t stream) {
		objForceExchanger->finalize();
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



	scheduler.addDependency("Сell-lists", {"Clear forces", "Halo init", "Object internal forces"}, {});

	scheduler.addDependency("Plugins: before forces", {"Internal forces", "Halo forces", "Object internal forces"}, {});
	scheduler.addDependency("Internal forces", {}, {"Clear forces"});
	scheduler.addDependency("Plugins: serialize and send", {}, {"Internal forces"});
	scheduler.addDependency("Halo init", {"Internal forces"}, {});
	scheduler.addDependency("Halo finalize", {}, {"Halo init"});
	scheduler.addDependency("Halo forces", {}, {"Halo finalize"});
	scheduler.addDependency("Accumulate forces", {"Integration"}, {"Halo forces", "Internal forces"});
	scheduler.addDependency("Plugins: before integration", {"Integration"}, {});

	scheduler.addDependency("Object halo init", {}, {"Integrate"});
	scheduler.addDependency("Object halo finalize", {}, {"Object halo init"});

	scheduler.addDependency("Object bounce", {}, {"Object halo finalize", "Object integration", "Integration"});
	scheduler.addDependency("Obj forces exchange: init", {"Redistribute init"}, {"Object bounce", "Object internal forces"});
	scheduler.addDependency("Obj forces exchange: finalize", {}, {"Obj forces exchange: init"});

	scheduler.addDependency("Plugins: after integration", {}, {"Integration", "Wall bounce", "Send obj forces"});
	scheduler.addDependency("Redistribute init", {}, {"Integration", "Wall bounce", "Send obj forces", "Plugins: after integration"});
	scheduler.addDependency("Redistribute finalize", {}, {"Redistribute init"});

	scheduler.setHighPriority("Object internal forces");
	scheduler.setHighPriority("Object halo init");
	scheduler.setHighPriority("Object halo finalize");
	scheduler.setHighPriority("Object halo forces");
	scheduler.setHighPriority("Object accumulate forces");
	scheduler.setHighPriority("Object integrate");

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



