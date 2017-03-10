#pragma once

#include <core/datatypes.h>
#include <core/components.h>
#include <core/wall.h>
#include <core/interactions.h>
#include <plugins/plugin.h>

#include <vector>
#include <string>
#include <map>
#include <mpi.h>

class HaloExchanger;
class Redistributor;
class Wall;
class ParticleVector;
class ObjectVector;
class CellList;

class Simulation
{
	int rank;
	int3 rank3D;
	MPI_Comm cartComm;
	MPI_Comm interComm;

public:
	int3 nranks3D;
	float3 globalDomainSize, subDomainSize, subDomainStart;

private:
	double currentTime;
	int currentStep;
	cudaStream_t defStream;

private:

	HaloExchanger* halo;
	Redistributor* redistributor;

	std::map<std::string, int> pvIdMap;
	std::vector<ParticleVector*> particleVectors;

	std::map<std::string, Interaction*> interactionMap;
	std::map<std::string, Integrator*>  integratorMap;
	std::map<std::string, Wall*>        wallMap;

//	std::vector<Interaction*>    interactions;

	struct RobustFloatLess
	{
		bool operator() (const float& a, const float& b)
		{
			const float eps = 1e-6;
			if (fabs(a - b) < eps) return false;
			return a < b;
		}
	};

	std::multimap< float, std::function<void(InteractionType, float, cudaStream_t)>, RobustFloatLess > forceCallers;

	std::vector< std::map<float, CellList*, RobustFloatLess> > cellListMaps;
	std::vector<Integrator*>     integrators;


	std::vector<SimulationPlugin*> plugins;

public:
	Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm);

	void registerParticleVector(ParticleVector* pv, InitialConditions* ic);
	void registerObjectVector  (ObjectVector* ov);
	void registerWall          (Wall* wall);

	void registerInteraction   (Interaction* interaction);
	void registerIntegrator    (Integrator* integrator);

	void setIntegrator (std::string pvName, std::string integratorName);
	void setInteraction(std::string pv1Name, std::string pv2Name, std::string interactionName);

	void registerPlugin(SimulationPlugin* plugin);

	void init();
	void run(int nsteps);
	void createWalls();
	void finalize();

	std::vector<int> getWallCreationSteps() const;

	const std::map<std::string, int>&   getPvIdMap() const { return pvIdMap; }
	const std::vector<ParticleVector*>& getParticleVectors() const { return particleVectors; }
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
