#pragma once

#include <core/logger.h>
#include <core/utils/cuda_common.h>

class Simulation;
class Postprocess;
class SimulationPlugin;
class PostprocessPlugin;

class uDeviceX
{
private:
	int pluginId = 0;
	int computeTask;
	bool noPostprocess;

	void sayHello();

public:
	Simulation* sim;
	Postprocess* post;

	uDeviceX(int3 nranks3D, float3 globalDomainSize,
			Logger& logger, std::string logFileName, int verbosity=3);

	bool isComputeTask();
	void registerJointPlugins(SimulationPlugin* simPl, PostprocessPlugin* postPl);
	void run(int niters);
};
