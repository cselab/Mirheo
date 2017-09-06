#pragma once

#include <core/logger.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/wall.h>
#include <core/interactions/interface.h>
#include <core/integrate.h>
#include <core/initial_conditions.h>
#include <core/task_scheduler.h>
#include <core/mpi/api.h>
#include <plugins/plugin.h>


#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <mpi.h>

class Simulation
{
public:
	int3 nranks3D;
	float3 globalDomainSize, globalDomainStart, localDomainSize;

private:
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

	std::map<std::string, Interaction*> interactionMap;
	std::map<std::string, Integrator*>  integratorMap;
	std::map<std::string, Wall*>        wallMap;

	std::vector<std::tuple<float, ParticleVector*, ParticleVector*, Interaction*>> interactionPrototypes;
	std::vector<std::tuple<Wall*, ParticleVector*, float>> wallProtorypes;

	std::vector<std::function<void(float, cudaStream_t)>> regularInteractions, haloInteractions;
	std::map<ParticleVector*, std::vector<CellList*>> cellListMap;

	std::vector<Integrator*>     integrators;
	std::vector<SimulationPlugin*> plugins;

	void assemble();

public:
	Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm);

	void registerParticleVector(ParticleVector* pv, InitialConditions* ic);
	void registerObjectVector  (ObjectVector* ov);
	void registerWall          (Wall* wall, bool addCorrespondingPV);

	void registerInteraction   (Interaction* interaction);
	void registerIntegrator    (Integrator* integrator);

	void setIntegrator (std::string pvName, std::string integratorName);
	void setInteraction(std::string pv1Name, std::string pv2Name, std::string interactionName);

	void registerPlugin(SimulationPlugin* plugin);

	void init();
	void run(int nsteps);
	void createWalls();
	void finalize();


	const std::map<std::string, int>&   getPvIdMap() const { return pvIdMap; }
	const std::vector<ParticleVector*>& getParticleVectors() const { return particleVectors; }

	ParticleVector* getPVbyName(std::string name) const
	{
		auto pvIt = pvIdMap.find(name);
		return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second] : nullptr;
	}

	MPI_Comm getCartComm() const { return cartComm; }

};

class Postprocess
{
private:
	MPI_Comm comm;
	MPI_Comm interComm;
	std::vector<PostprocessPlugin*> plugins;
	std::vector<MPI_Request> requests;

public:
	Postprocess(MPI_Comm& comm, MPI_Comm& interComm);
	void registerPlugin(PostprocessPlugin* plugin);
	void run();
};

class uDeviceX
{
	int pluginId = 0;
	int computeTask;
	bool noPostprocess;

public:
	Simulation* sim;
	Postprocess* post;

	uDeviceX(int argc, char** argv, int3 nranks3D, float3 globalDomainSize,
			Logger& logger, std::string logFileName, int verbosity=3, bool noPostprocess = false);
	bool isComputeTask();
	void registerJointPlugins(SimulationPlugin* simPl, PostprocessPlugin* postPl);
	void run(int niters);
};
