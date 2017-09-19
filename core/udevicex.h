#pragma once

#include <core/simulation.h>
#include <core/postproc.h>
#include <plugins/interface.h>

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
