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

#include "domain.h"

#include <tuple>
#include <vector>
#include <string>
#include <map>

// Some forward declarations
class ParticleVector;
class ObjectVector;
class CellList;
class TaskScheduler;


class ParticleHaloExchanger;
class ParticleRedistributor;

class ObjectHaloExchanger;
class ObjectRedistributor;
class ObjectForcesReverseExchanger;


class Simulation
{
public:
	int3 nranks3D;
	int3 rank3D;

	MPI_Comm cartComm;
	MPI_Comm interComm;

	DomainInfo domain;

	Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm);

	void registerParticleVector         (std::unique_ptr<ParticleVector> pv, std::unique_ptr<InitialConditions> ic, int checkpointEvery);
	void registerWall                   (std::unique_ptr<Wall> wall, int checkEvery=0);
	void registerInteraction            (std::unique_ptr<Interaction> interaction);
	void registerIntegrator             (std::unique_ptr<Integrator> integrator);
	void registerBouncer                (std::unique_ptr<Bouncer> bouncer);
	void registerPlugin                 (std::unique_ptr<SimulationPlugin> plugin);
	void registerObjectBelongingChecker (std::unique_ptr<ObjectBelongingChecker> checker);


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

	std::vector<ParticleVector*> getParticleVectors() const
	{
		std::vector<ParticleVector*> res;
		for (auto& pv : particleVectors)
			res.push_back(pv.get());

		return res;
	}

	ParticleVector* getPVbyName(std::string name) const
	{
		auto pvIt = pvIdMap.find(name);
		return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second].get() : nullptr;
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
			return clvecIt->second[0].get();
	}

	MPI_Comm getCartComm() const { return cartComm; }

private:
	const float rcTolerance = 1e-5;

	std::string restartFolder;

	float dt;
	int rank;

	double currentTime;
	int currentStep;

	std::unique_ptr<TaskScheduler> scheduler;

	std::unique_ptr<ParticleHaloExchanger> halo;
	std::unique_ptr<ParticleRedistributor> redistributor;

	std::unique_ptr<ObjectHaloExchanger> objHalo;
	std::unique_ptr<ObjectRedistributor> objRedistibutor;
	std::unique_ptr<ObjectForcesReverseExchanger> objHaloForces;

	std::map<std::string, int> pvIdMap;
	std::vector< std::unique_ptr<ParticleVector> > particleVectors;
	std::vector< ObjectVector* >   objectVectors;

	std::map< std::string, std::unique_ptr<Bouncer> >                bouncerMap;
	std::map< std::string, std::unique_ptr<Integrator> >             integratorMap;
	std::map< std::string, std::unique_ptr<Interaction> >            interactionMap;
	std::map< std::string, std::unique_ptr<Wall> >                   wallMap;
	std::map< std::string, std::unique_ptr<ObjectBelongingChecker> > belongingCheckerMap;

	std::map<ParticleVector*, std::vector< std::unique_ptr<CellList> >> cellListMap;

	std::vector<std::tuple<float, ParticleVector*, ParticleVector*, Interaction*>> interactionPrototypes;
	std::vector<std::tuple<Wall*, ParticleVector*>> wallPrototypes;
	std::vector<std::tuple<Wall*, int>> checkWallPrototypes;
	std::vector<std::tuple<Bouncer*, ParticleVector*>> bouncerPrototypes;
	std::vector<std::tuple<ObjectBelongingChecker*, ParticleVector*, ParticleVector*, int>> belongingCorrectionPrototypes;
	std::vector<std::tuple<ObjectBelongingChecker*, ParticleVector*, ParticleVector*, ParticleVector*>> splitterPrototypes;


	std::vector<std::function<void(float, cudaStream_t)>> regularInteractions, haloInteractions;
	std::vector<std::function<void(float, cudaStream_t)>> integratorsStage1, integratorsStage2;
	std::vector<std::function<void(float, cudaStream_t)>> regularBouncers, haloBouncers;

	std::vector< std::unique_ptr<SimulationPlugin> > plugins;

	void prepareCellLists();
	void prepareInteractions();
	void prepareBouncers();
	void prepareWalls();
	void execSplitters();

	void assemble();
};






