#pragma once

#include <core/logger.h>
#include <core/utils/pytypes.h>

class Simulation;
class Postprocess;
class SimulationPlugin;
class PostprocessPlugin;

class ParticleVector;
class InitialConditions;
class Integrator;
class Interaction;

class uDeviceX
{
public:
    uDeviceX(pyint3 nranks3D, pyfloat3 globalDomainSize,
             std::string logFileName, int verbosity, bool gpuAwareMPI);

    bool isComputeTask();
    void run(int niters);
    
    void registerParticleVector         (ParticleVector* pv, InitialConditions* ic, int checkpointEvery);
    void registerInteraction            (Interaction* interaction);
    void registerIntegrator             (Integrator* integrator);
//     void registerWall                   (PyWall wall, int checkEvery);
//     void registerBouncer                (PyBouncer bouncer);
//     void registerPlugin                 (PyPlugin plugin);
//     void registerObjectBelongingChecker (PyObjectBelongingChecker checker);
// 
    void setIntegrator             (Integrator* integrator,  ParticleVector* pv);
    void setInteraction            (Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2);
//     void setBouncer                (std::string bouncerName,     std::string objName, std::string pvName);
//     void setWallBounce             (std::string wallName,        std::string pvName);
//     void setObjectBelongingChecker (std::string checkerName,     std::string objName);
        
    ~uDeviceX();

private:
    std::unique_ptr<Simulation> sim;
    std::unique_ptr<Postprocess> post;
    
    int computeTask;
    bool noPostprocess;
    
    //void registerPlugins( std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> > plugins );
    void sayHello();
};
