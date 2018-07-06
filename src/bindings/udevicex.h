#pragma once

#include <core/logger.h>

#include "integrators.h"
#include "particle_vectors.h"
#include "initial_conditions.h"

class Simulation;
class Postprocess;
class SimulationPlugin;
class PostprocessPlugin;

class uDeviceX
{
public:
	uDeviceX(std::tuple<int, int, int> nranks3D, std::tuple<float, float, float> globalDomainSize,
             std::string logFileName, int verbosity, bool gpuAwareMPI);

	bool isComputeTask();
	void run(int niters);
    
    void registerParticleVector         (PyParticleVector* pv, PyInitialConditions* ic, int checkpointEvery);
// 	void registerWall                   (PyWall wall, int checkEvery);
// 	void registerInteraction            (PyInteraction interaction);
 	void registerIntegrator             (PyIntegrator* integrator);
// 	void registerBouncer                (PyBouncer bouncer);
// 	void registerPlugin                 (PyPlugin plugin);
// 	void registerObjectBelongingChecker (PyObjectBelongingChecker checker);
// 
// 	void setIntegrator             (std::string integratorName,  std::string pvName);
// 	void setInteraction            (std::string interactionName, std::string pv1Name, std::string pv2Name);
// 	void setBouncer                (std::string bouncerName,     std::string objName, std::string pvName);
// 	void setWallBounce             (std::string wallName,        std::string pvName);
// 	void setObjectBelongingChecker (std::string checkerName,     std::string objName);
        
	~uDeviceX();

private:
    std::unique_ptr<Simulation> sim;
	std::unique_ptr<Postprocess> post;
    
	int computeTask;
	bool noPostprocess;
    
    void registerPlugins( std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> > plugins );
	void sayHello();
};
