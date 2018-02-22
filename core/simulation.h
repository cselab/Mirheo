#pragma once

#include <core/logger.h>
#include <core/datatypes.h>
#include <core/containers.h>

#include <core/bouncers/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>

#include <core/task_scheduler.h>
#include <core/mpi/api.h>

#include "domain.h"

#include <tuple>
#include <vector>
#include <string>
#include <map>

// Some forward declarations
class ParticleVector;
class ObjectVector;
class CellList;

class Simulation
{
public:
	int3 nranks3D;
	int3 rank3D;

	MPI_Comm cartComm;
	MPI_Comm interComm;

	DomainInfo domain;

	Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm);

	void registerParticleVector         (ParticleVector* pv, InitialConditions* ic, int checkpointEvery);
	void registerWall                   (Wall* wall, int checkEvery=0);
	void registerInteraction            (Interaction* interaction);
	void registerIntegrator             (Integrator* integrator);
	void registerBouncer                (Bouncer* bouncer);
	void registerPlugin                 (SimulationPlugin* plugin);
	void registerObjectBelongingChecker (ObjectBelongingChecker* checker);


	void setIntegrator             (std::string integratorName,  std::string pvName);
	void setInteraction            (std::string interactionName, std::string pv1Name, std::string pv2Name);
	void setBouncer                (std::string bouncerName,     std::string objName, std::string pvName);
	void setWallBounce             (std::string wallName,        std::string pvName);
	void setObjectBelongingChecker (std::string checkerName,     std::string objName);


	void applyObjectBelongingChecker(std::string checkerName,
			std::string source, std::string inside, std::string outside, int checkEvery);


	void init();
	void run(int nsteps);
	void finalize();

	const std::vector<ParticleVector*>& getParticleVectors() const { return particleVectors; }

	ParticleVector* getPVbyName(std::string name) const
	{
		auto pvIt = pvIdMap.find(name);
		return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second] : nullptr;
	}

	ParticleVector* getPVbyNameOrDie(std::string name) const
	{
		auto pv = getPVbyName(name);
		if (pv == nullptr)
			die("No such particle vector: %s", name.c_str());
		return pv;
	}

	ObjectVector* getOVbyNameOrDie(std::string name) const
	{
		auto pv = getPVbyName(name);
		auto ov = dynamic_cast<ObjectVector*>(pv);
		if (pv == nullptr)
			die("No such particle vector: %s", name.c_str());
		return ov;
	}

	CellList* gelCellList(ParticleVector* pv) const
	{
		auto clvecIt = cellListMap.find(pv);
		if (clvecIt == cellListMap.end())
			die("Particle Vector '%s' is not registered or broken", pv->name.c_str());

		if (clvecIt->second.size() == 0)
			return nullptr;
		else
			return clvecIt->second[0];
	}

	MPI_Comm getCartComm() const { return cartComm; }

private:
	const float rcTolerance = 1e-5;

	std::string restartFolder;

	float dt;
	int rank;

	double currentTime;
	int currentStep;

	TaskScheduler scheduler;

	ParticleHaloExchanger* halo;
	ParticleRedistributor* redistributor;

	ObjectHaloExchanger* objHalo;
	ObjectRedistributor* objRedistibutor;
	ObjectForcesReverseExchanger* objHaloForces;

	std::map<std::string, int> pvIdMap;
	std::vector<ParticleVector*> particleVectors;
	std::vector<ObjectVector*>   objectVectors;

	std::map<std::string, Bouncer*>                bouncerMap;
	std::map<std::string, Integrator*>             integratorMap;
	std::map<std::string, Interaction*>            interactionMap;
	std::map<std::string, Wall*>                   wallMap;
	std::map<std::string, ObjectBelongingChecker*> belongingCheckerMap;

	std::map<ParticleVector*, std::vector<CellList*>> cellListMap;

	std::vector<std::tuple<float, ParticleVector*, ParticleVector*, Interaction*>> interactionPrototypes;
	std::vector<std::tuple<Wall*, ParticleVector*>> wallPrototypes;
	std::vector<std::tuple<Wall*, int>> checkWallPrototypes;
	std::vector<std::tuple<Bouncer*, ParticleVector*>> bouncerPrototypes;
	std::vector<std::tuple<ObjectBelongingChecker*, ParticleVector*, int>> belongingCheckerPrototypes;
	std::vector<std::tuple<ObjectBelongingChecker*, ParticleVector*, ParticleVector*, ParticleVector*>> splitterPrototypes;


	std::vector<std::function<void(float, cudaStream_t)>> regularInteractions, haloInteractions;
	std::vector<std::function<void(float, cudaStream_t)>> integratorsStage1, integratorsStage2;
	std::vector<std::function<void(float, cudaStream_t)>> regularBouncers, haloBouncers;

	std::vector<SimulationPlugin*> plugins;

	void prepareCellLists();
	void prepareInteractions();
	void prepareBouncers();
	void prepareWalls();
	void execSplitters();

	void assemble();
};






