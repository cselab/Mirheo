#pragma once

#include "datatypes.h"
#include "containers.h"
#include "components.h"
#include "logger.h"
#include "../plugins/plugin.h"

#include <vector>
#include <string>
#include <unordered_map>

class Simulation
{
	int3 nranks;
	int rank;
	int3 rank3D;
	float3 fullDomainSize, subDomainSize, subDomainStart;
	MPI_Comm cartComm;

private:

	std::unordered_map<std::string, int> pvMap;
	std::unordered_map<std::string, Interaction*> interactionMap;
	std::unordered_map<std::string, Integrator*>  integratorMap;

	std::vector<ParticleVector*> particleVectors;
	std::vector<Interaction*>    interactions;
	std::vector<Integrator*>     integrators;
	std::vector<CellList*>       cellLists;

	std::vector<std::vector<CellList*>> cellListTable;
	std::vector<std::vector< std::pair<Interaction*, CellList*> >> interactionTable;

	std::vector<SimulationPlugin*> plugins;

public:
	Simulation(int3 nranks, float3 domainSize, MPI_Comm& comm);

	void registerParticleVector(std::string name, ParticleVector* pv);
	void registerObjectVector  (std::string name, ObjectVector* ov);
	void registerInteraction   (std::string name, Interaction* interaction);
	void registerIntegrator    (std::string name, Integrator* integrator);
	void registerWall          (std::string name, Wall* wall);

	void registerPlugin(SimulationPlugin* plugin);

	void setIntegrator (std::string pvName, std::string integratorName);
	void setInteraction(std::string pv1Name, std::string pv2Name, std::string interactionName);

	void run(int nsteps);
};

class Postprocess
{
private:
	MPI_Comm comm;
	std::vector<PostprocessPlugin*> plugins;
	std::vector<MPI_Request> requests;

public:
	Postprocess(MPI_Comm& comm);
	void registerPlugin(PostprocessPlugin* plugin);
	void run();
};

class uDeviceX
{
	int pluginId = 0;

public:
	Simulation* sim;
	Postprocess* post;

	void registerJointPlugins();
	void run();
};
