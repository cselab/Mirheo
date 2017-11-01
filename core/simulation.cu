#include "simulation.h"

#include <algorithm>

Simulation::Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm) :
nranks3D(nranks3D), interComm(interComm), currentTime(0), currentStep(0)
{
	int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
	int periods[] = {1, 1, 1};
	int coords[3];

	MPI_Check( MPI_Comm_rank(comm, &rank) );
	MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
	MPI_Check( MPI_Cart_get(cartComm, 3, ranksArr, periods, coords) );
	rank3D = {coords[0], coords[1], coords[2]};

	domain.globalSize = globalDomainSize;
	domain.localSize = domain.globalSize / make_float3(nranks3D);
	domain.globalStart = {domain.localSize.x * coords[0], domain.localSize.y * coords[1], domain.localSize.z * coords[2]};

//	restartFolder  = "./restart/";
//	std::string command = "mkdir -p " + restartFolder;
//	if (rank == 0)
//	{
//		if ( system(command.c_str()) != 0 )
//		{
//			error("Could not create folder for restart files, will try to use ./");
//			restartFolder = "./";
//		}
//	}

	info("Simulation initialized, subdomain size is [%f %f %f], subdomain starts at [%f %f %f]",
			domain.localSize.x,  domain.localSize.y,  domain.localSize.z,
			domain.globalStart.x, domain.globalStart.y, domain.globalStart.z);
}

//================================================================================================
// Registration
//================================================================================================

void Simulation::registerParticleVector(ParticleVector* pv, InitialConditions* ic)
{
	std::string name = pv->name;

	if (name == "none" || name == "all" || name == "")
		die("Invalid name for a particle vector (reserved or empty): '%s'", name.c_str());

	particleVectors.push_back(pv);

	auto ov = dynamic_cast<ObjectVector*>(pv);
	if(ov != nullptr)
		objectVectors.push_back(ov);

	if (pvIdMap.find(name) != pvIdMap.end())
		die("More than one particle vector is called %s", name.c_str());

	pvIdMap[name] = particleVectors.size() - 1;

	if (ic != nullptr)
		ic->exec(cartComm, pv, domain, 0);
	else // TODO: get rid of this
		pv->domain = domain;
}

void Simulation::registerWall(Wall* wall, int every)
{
	std::string name = wall->name;

	if (wallMap.find(name) != wallMap.end())
		die("More than one wall is called %s", name.c_str());

	wallMap[name] = wall;
	checkWallPrototypes.push_back(std::make_tuple(wall, every));

	wall->setup(cartComm, domain);
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

void Simulation::registerObjectBelongingChecker(ObjectBelongingChecker* checker)
{
	std::string name = checker->name;
	if (belongingCheckerMap.find(name) != belongingCheckerMap.end())
		die("More than one splitter is called %s", name.c_str());

	belongingCheckerMap[name] = checker;
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
	auto integrator = integratorMap[integratorName];

	auto pv = getPVbyNameOrDie(pvName);

	integratorsStage1.push_back([integrator, pv] (float t, cudaStream_t stream) {
		integrator->stage1(pv, t, stream);
	});

	integratorsStage2.push_back([integrator, pv] (float t, cudaStream_t stream) {
		integrator->stage2(pv, t, stream);
	});
}

void Simulation::setInteraction(std::string interactionName, std::string pv1Name, std::string pv2Name)
{
	auto pv1 = getPVbyNameOrDie(pv1Name);
	auto pv2 = getPVbyNameOrDie(pv2Name);

	if (interactionMap.find(interactionName) == interactionMap.end())
		die("No such integrator: %s", interactionName.c_str());
	auto interaction = interactionMap[interactionName];


	float rc = interaction->rc;
	interactionPrototypes.push_back(std::make_tuple(rc, pv1, pv2, interaction));
}

void Simulation::setBouncer(std::string bouncerName, std::string objName, std::string pvName)
{
	auto pv = getPVbyNameOrDie(pvName);

	auto ov = dynamic_cast<ObjectVector*> (getPVbyName(objName));
	if (ov == nullptr)
		die("No such object vector: %s", objName.c_str());

	if (bouncerMap.find(bouncerName) == bouncerMap.end())
		die("No such bouncer: %s", bouncerName.c_str());
	auto bouncer = bouncerMap[bouncerName];

	bouncer->setup(ov);
	bouncerPrototypes.push_back(std::make_tuple(bouncer, pv));
}

void Simulation::setWallBounce(std::string wallName, std::string pvName)
{
	auto pv = getPVbyNameOrDie(pvName);

	if (wallMap.find(wallName) == wallMap.end())
		die("No such wall: %s", wallName.c_str());
	auto wall = wallMap[wallName];

	wallPrototypes.push_back( std::make_tuple(wall, pv) );
}

void Simulation::setObjectBelongingChecker(std::string checkerName, std::string objName)
{
	auto ov = dynamic_cast<ObjectVector*>(getPVbyNameOrDie(objName));
	if (ov == nullptr)
		die("No such object vector %s", objName.c_str());

	if (belongingCheckerMap.find(checkerName) == belongingCheckerMap.end())
		die("No such belonging checker: %s", checkerName.c_str());
	auto checker = belongingCheckerMap[checkerName];

	// TODO: do this normal'no blyat!
	checker->setup(ov);
}

//
//
//

void Simulation::applyObjectBelongingChecker(std::string checkerName,
			std::string source, std::string inside, std::string outside, int checkEvery)
{
	auto pvSource = getPVbyNameOrDie(source);

	if (inside == outside)
		die("Splitting into same pvs: %s into %s %s",
				source.c_str(), inside.c_str(), outside.c_str());

	if (source != inside && source != outside)
		die("At least one of the split destinations should be the same as source: %s into %s %s",
				source.c_str(), inside.c_str(), outside.c_str());

	if (belongingCheckerMap.find(checkerName) == belongingCheckerMap.end())
		die("No such belonging checker: %s", checkerName.c_str());

	if (getPVbyName(inside) != nullptr && inside != source)
		die("Cannot split into existing particle vector: %s into %s %s",
				source.c_str(), inside.c_str(), outside.c_str());

	if (getPVbyName(outside) != nullptr && outside != source)
		die("Cannot split into existing particle vector: %s into %s %s",
				source.c_str(), inside.c_str(), outside.c_str());


	auto checker = belongingCheckerMap[checkerName];

	ParticleVector *pvInside  = getPVbyName(inside);
	ParticleVector *pvOutside = getPVbyName(outside);

	if (inside != "none" && pvInside == nullptr)
	{
		pvInside = new ParticleVector(inside, pvSource->mass);
		registerParticleVector(pvInside, nullptr);
	}

	if (outside != "none" && pvOutside == nullptr)
	{
		pvOutside = new ParticleVector(outside, pvSource->mass);
		registerParticleVector(pvOutside, nullptr);
	}

	splitterPrototypes.push_back(std::make_tuple(checker, pvSource, pvInside, pvOutside));

	if (pvInside != nullptr)
		belongingCheckerPrototypes.push_back(std::make_tuple(checker, pvInside,  checkEvery));

	if (pvOutside != nullptr)
		belongingCheckerPrototypes.push_back(std::make_tuple(checker, pvOutside, checkEvery));
}


void Simulation::prepareCellLists()
{
	info("Preparing cell-lists");

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
			cellListMap[cutoffs.first].push_back(primary ?
					new PrimaryCellList(cutoffs.first, rc, domain.localSize) :
					new CellList       (cutoffs.first, rc, domain.localSize));
			primary = false;
		}
	}
}

void Simulation::prepareInteractions()
{
	info("Preparing interactions");

	for (auto prototype : interactionPrototypes)
	{
		auto  rc = std::get<0>(prototype);
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
	info("Preparing object bouncers");

	for (auto prototype : bouncerPrototypes)
	{
		auto bouncer = std::get<0>(prototype);
		auto pv = std::get<1>(prototype);

		auto& clVec = cellListMap[pv];

		if (clVec.empty()) continue;

		CellList *cl = clVec[0];

		regularBouncers.push_back([bouncer, pv, cl] (float dt, cudaStream_t stream) {
			bouncer->bounceLocal(pv, cl, dt, stream);
		});

		haloBouncers.   push_back([bouncer, pv, cl] (float dt, cudaStream_t stream) {
			bouncer->bounceHalo (pv, cl, dt, stream);
		});
	}
}

void Simulation::prepareWalls()
{
	info("Preparing walls");

	for (auto prototype : wallPrototypes)
	{
		auto wall  = std::get<0>(prototype);
		auto pv    = std::get<1>(prototype);

		auto& clVec = cellListMap[pv];

		if (clVec.empty()) continue;

		CellList *cl = clVec[0];

		wall->attach(pv, cl);
		wall->removeInner(pv);
	}
}

void Simulation::execSplitters()
{
	info("Splitting particle vectors with respect to object belonging");

	for (auto prototype : splitterPrototypes)
	{
		auto checker = std::get<0>(prototype);
		auto src     = std::get<1>(prototype);
		auto inside  = std::get<2>(prototype);
		auto outside = std::get<3>(prototype);

		checker->splitByBelonging(src, inside, outside, 0);
	}
}

void Simulation::init()
{
	info("Simulation initiated");

	prepareCellLists();

	prepareInteractions();
	prepareBouncers();
	prepareWalls();

	CUDA_Check( cudaDeviceSynchronize() );

	info("Preparing plugins");
	for (auto& pl : plugins)
	{
		debug("Setup and handshake of plugin %s", pl->name.c_str());
		pl->setup(this, cartComm, interComm);
		pl->handshake();
	}

	halo = new ParticleHaloExchanger(cartComm);
	redistributor = new ParticleRedistributor(cartComm);

	objHalo = new ObjectHaloExchanger(cartComm);
	objRedistibutor = new ObjectRedistributor(cartComm);
	objHaloForces = new ObjectForcesReverseExchanger(cartComm, objHalo);

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

				objHalo->        attach(ov, cl->rc);
				objHaloForces->  attach(ov);
				objRedistibutor->attach(ov, cl->rc);
			}

	assemble();
}

void Simulation::assemble()
{
	// XXX: different dt not implemented
	dt = 1.0;
	for (auto& integr : integratorMap)
		dt = min(dt, integr.second->dt);


	scheduler.addTask("Сell-lists", [&] (cudaStream_t stream) {
		for (auto& clVec : cellListMap)
			for (auto cl : clVec.second)
				cl->build(stream);
	});

	// Only particle forces, not object ones here
	scheduler.addTask("Clear forces", [&] (cudaStream_t stream) {
		for (auto pv : particleVectors)
		{
			auto& clVec = cellListMap[pv];
			for (auto cl : clVec)
				cl->forces->clear(stream);
		}
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
		halo->finalize(stream);
	});

	scheduler.addTask("Halo forces", [&] (cudaStream_t stream) {
		for (auto& inter : haloInteractions)
			inter(currentTime, stream);
	});

	scheduler.addTask("Accumulate forces", [&] (cudaStream_t stream) {
		for (auto& clVec : cellListMap)
			for (auto cl : clVec.second)
				cl->addForces(stream);
	});

	scheduler.addTask("Plugins: before integration", [&] (cudaStream_t stream) {
		for (auto& pl : plugins)
			pl->beforeIntegration(stream);
	});

	scheduler.addTask("Integration", [&] (cudaStream_t stream) {
		for (auto& integrator : integratorsStage2)
			integrator(currentTime, stream);
	});


	scheduler.addTask("Object halo init", [&] (cudaStream_t stream) {
		objHalo->init(stream);
	});

	scheduler.addTask("Object halo finalize", [&] (cudaStream_t stream) {
		objHalo->finalize(stream);
	});

	scheduler.addTask("Clear obj halo forces", [&] (cudaStream_t stream) {
		for (auto& ov : objectVectors)
			ov->halo()->forces.clear(stream);
	});

	// As there are no primary cell-lists for objects
	// we need to separately clear real obj forces and forces in the cell-lists
	scheduler.addTask("Clear obj local forces", [&] (cudaStream_t stream) {
		for (auto ov : objectVectors)
		{
			ov->local()->forces.clear(stream);

			auto& clVec = cellListMap[ov];
			for (auto cl : clVec)
				cl->forces->clear(stream);
		}
	});

	scheduler.addTask("Object bounce", [&] (cudaStream_t stream) {
		for (auto& bouncer : regularBouncers)
			bouncer(dt, stream);

		for (auto& bouncer : haloBouncers)
			bouncer(dt, stream);
	});

	for (auto& prototype : belongingCheckerPrototypes)
	{
		auto checker = std::get<0>(prototype);
		auto pv      = std::get<1>(prototype);
		auto every   = std::get<2>(prototype);

		if (every > 0)
		{
			auto clVec = cellListMap[pv];
			if (clVec.size() == 0)
				die("Unable to check belonging of a PV without a valid cell-list");
			auto cl = clVec[0];

			scheduler.addTask("Bounce check",
					[checker, pv, cl] (cudaStream_t stream) { checker->checkInner(pv, cl, stream); },
					every);
		}
	}

	scheduler.addTask("Obj forces exchange: init", [&] (cudaStream_t stream) {
		objHaloForces->init(stream);
	});

	scheduler.addTask("Obj forces exchange: finalize", [&] (cudaStream_t stream) {
		objHaloForces->finalize(stream);
	});

	scheduler.addTask("Wall bounce", [&] (cudaStream_t stream) {
		for (auto& wall : wallMap)
			wall.second->bounce(dt, stream);
	});

	for (auto& prototype : checkWallPrototypes)
	{
		auto wall  = std::get<0>(prototype);
		auto every = std::get<1>(prototype);

		if (every > 0)
			scheduler.addTask("Wall check", [&, wall] (cudaStream_t stream) { wall->check(stream); }, every);
	}

	scheduler.addTask("Plugins: after integration", [&] (cudaStream_t stream) {
		for (auto pl : plugins)
			pl->afterIntegration(stream);
	});

	scheduler.addTask("Redistribute init", [&] (cudaStream_t stream) {
		redistributor->init(stream);
	});

	scheduler.addTask("Redistribute finalize", [&] (cudaStream_t stream) {
		redistributor->finalize(stream);
	});

	scheduler.addTask("Object extents", [&] (cudaStream_t stream) {
		for (auto& ov : objectVectors)
			ov->findExtentAndCOM(stream);
	});

	// This one should be executed after halo and redist
	// as bounces and other things rely on correct extent
	scheduler.addTask("Object extents 2", [&] (cudaStream_t stream) {
		for (auto& ov : objectVectors)
			ov->findExtentAndCOM(stream);
	});

	scheduler.addTask("Object redistribute init", [&] (cudaStream_t stream) {
		objRedistibutor->init(stream);
	});

	scheduler.addTask("Object redistribute finalize", [&] (cudaStream_t stream) {
		objRedistibutor->finalize(stream);
	});



	scheduler.addDependency("Сell-lists", {"Clear forces"}, {});

	scheduler.addDependency("Plugins: before forces", {"Internal forces", "Halo forces"}, {"Clear forces"});
	scheduler.addDependency("Plugins: serialize and send", {"Redistribute init", "Object redistribute init"}, {"Plugins: before forces"});

	scheduler.addDependency("Internal forces", {}, {"Clear forces"});

	scheduler.addDependency("Clear obj halo forces", {"Object bounce"}, {"Object halo finalize"});

	scheduler.addDependency("Obj forces exchange: init", {}, {"Halo forces"});
	scheduler.addDependency("Obj forces exchange: finalize", {"Accumulate forces"}, {"Obj forces exchange: init"});

	scheduler.addDependency("Halo init", {}, {"Plugins: before forces"});
	scheduler.addDependency("Halo finalize", {}, {"Halo init"});
	scheduler.addDependency("Halo forces", {}, {"Halo finalize"});

	scheduler.addDependency("Accumulate forces", {"Integration"}, {"Halo forces", "Internal forces"});
	scheduler.addDependency("Plugins: before integration", {"Integration"}, {"Accumulate forces"});
	scheduler.addDependency("Wall bounce", {}, {"Integration"});
	scheduler.addDependency("Wall check", {}, {"Wall bounce"});

	scheduler.addDependency("Object halo init", {}, {"Integration", "Object redistribute finalize"});
	scheduler.addDependency("Object halo finalize", {}, {"Object halo init"});

	scheduler.addDependency("Object bounce", {}, {"Integration", "Object halo finalize", "Clear obj local forces"});
	scheduler.addDependency("Bounce check", {"Redistribute init"}, {"Object bounce"});

	scheduler.addDependency("Plugins: after integration", {"Object bounce"}, {"Integration", "Wall bounce"});

	scheduler.addDependency("Redistribute init", {}, {"Integration", "Wall bounce", "Object bounce", "Plugins: after integration"});
	scheduler.addDependency("Redistribute finalize", {}, {"Redistribute init"});

	scheduler.addDependency("Object extents", {"Object redistribute init", "Object halo init"}, {"Plugins: after integration"});
	scheduler.addDependency("Object extents 2", {"Object bounce"}, {"Object redistribute finalize", "Object halo finalize"});
	scheduler.addDependency("Object redistribute init", {}, {"Integration", "Wall bounce", "Obj forces exchange: finalize", "Plugins: after integration"});
	scheduler.addDependency("Object redistribute finalize", {}, {"Object redistribute init"});
	scheduler.addDependency("Clear obj local forces", {}, {"Integration"});

	scheduler.setHighPriority("Obj forces exchange: init");
	scheduler.setHighPriority("Halo init");
	//scheduler.setHighPriority("Halo finalize");
	scheduler.setHighPriority("Halo forces");
	scheduler.setHighPriority("Plugins: serialize and send");

	scheduler.compile();
}

void Simulation::run(int nsteps)
{
	int begin = currentStep, end = currentStep + nsteps;

	// Initial preparation
	scheduler.forceExec("Object extents");
	scheduler.forceExec("Object halo init");
	scheduler.forceExec("Object halo finalize");
	scheduler.forceExec("Clear obj halo forces");
	scheduler.forceExec("Clear obj local forces");

	// Halo extents
	scheduler.forceExec("Object extents");
	execSplitters();

	info("Will run %d iterations now", nsteps);


	for (currentStep = begin; currentStep < end; currentStep++)
	{
		info("===============================================================================\n"
				"Timestep: %d, simulation time: %f", currentStep, currentTime);

		scheduler.run();

		currentTime += dt;
	}

	// Finish the redistribution by rebuilding the cell-lists
	scheduler.forceExec("Сell-lists");

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



