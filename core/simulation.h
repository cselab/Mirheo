#pragma once

#include <core/logger.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/bouncers/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>

#include <core/task_scheduler.h>
#include <core/mpi/api.h>
#include <plugins/interface.h>


#include <tuple>
#include <vector>
#include <string>
#include <map>

class Simulation
{
public:
	int3 nranks3D;
	float3 globalDomainSize, globalDomainStart, localDomainSize;

private:
	const float rcTolerance = 1e-5;

	std::string restartFolder;

	float dt;
	int rank;
	int3 rank3D;
	MPI_Comm cartComm;
	MPI_Comm interComm;

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

	std::map<std::string, Bouncer*>     bouncerMap;
	std::map<std::string, Integrator*>  integratorMap;
	std::map<std::string, Interaction*> interactionMap;
	std::map<std::string, Wall*>        wallMap;

	std::map<ParticleVector*, std::vector<CellList*>> cellListMap;

	std::vector<std::tuple<float, ParticleVector*, ParticleVector*, Interaction*>> interactionPrototypes;
	std::vector<std::pair<Wall*,   ParticleVector*>> wallPrototypes;
	std::vector<std::tuple<Bouncer*, ObjectVector*, ParticleVector*>> bouncerPrototypes;

	std::vector<std::function<void(float, cudaStream_t)>> regularInteractions, haloInteractions;
	std::vector<std::function<void(cudaStream_t)>> integratorsStage1, integratorsStage2;
	std::vector<std::function<void(float, cudaStream_t)>> regularBouncers, haloBouncers;

	std::vector<SimulationPlugin*> plugins;

	void prepareCellLists();
	void prepareInteractions();
	void prepareBouncers();
	void prepareWalls();

	void assemble();

public:
	Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm);

	void registerParticleVector(ParticleVector* pv, InitialConditions* ic);
	void registerWall          (Wall* wall);
	void registerInteraction   (Interaction* interaction);
	void registerIntegrator    (Integrator*  integrator);
	void registerBouncer       (Bouncer*     integrator);

	void setIntegrator (std::string integratorName,  std::string pvName);
	void setInteraction(std::string interactionName, std::string pv1Name, std::string pv2Name);
	void setBouncer    (std::string bouncerName,     std::string objName, std::string pvName);
	void setWallBounce (std::string wallName,        std::string pvName);

	void registerPlugin(SimulationPlugin* plugin);

	void init();
	void run(int nsteps);
	void createWalls();
	void finalize();


	const std::vector<ParticleVector*>& getParticleVectors() const { return particleVectors; }

	ParticleVector* getPVbyName(std::string name) const
	{
		auto pvIt = pvIdMap.find(name);
		return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second] : nullptr;
	}

	MPI_Comm getCartComm() const { return cartComm; }
};






