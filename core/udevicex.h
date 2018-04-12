#pragma once

#include <core/logger.h>
#include <core/utils/cuda_common.h>

class Simulation;
class Postprocess;
class SimulationPlugin;
class PostprocessPlugin;

class uDeviceX
{
public:
	Simulation* sim;
	Postprocess* post;

	uDeviceX(int3 nranks3D, float3 globalDomainSize,
			Logger& logger, std::string logFileName, int verbosity=3);

	bool isComputeTask();
	void run(int niters);
	void registerPlugins( std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> > plugins );

private:
	int computeTask;
	bool noPostprocess;

	void sayHello();
};
