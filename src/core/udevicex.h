#pragma once

#include <core/logger.h>
#include <core/utils/pytypes.h>

class Simulation;
class Postprocess;


class ParticleVector;
class ObjectVector;
class InitialConditions;
class Integrator;
class Interaction;
class ObjectBelongingChecker;
class Bouncer;
class Wall;
class SimulationPlugin;
class PostprocessPlugin;

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
    void registerWall                   (Wall* wall, int checkEvery=0);
    void registerBouncer                (Bouncer* bouncer);
    void registerPlugin                 (std::pair<SimulationPlugin*, PostprocessPlugin*> plugin);
    void registerObjectBelongingChecker (ObjectBelongingChecker* checker, ObjectVector* ov);
 
    void setIntegrator  (Integrator* integrator,  ParticleVector* pv);
    void setInteraction (Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2);
    void setBouncer     (Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv);
    void setWallBounce  (Wall* wall, ParticleVector* pv);
    
    ParticleVector* applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                ParticleVector* pv,
                                                int checkEvery,
                                                std::string inside = "",
                                                std::string outside = "");
        
    ~uDeviceX();

private:
    std::unique_ptr<Simulation> sim;
    std::unique_ptr<Postprocess> post;
    
    int computeTask;
    bool noPostprocess;
    
    //void registerPlugins( std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> > plugins );
    void sayHello();
};
