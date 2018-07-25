#pragma once

#include <core/logger.h>
#include <core/utils/pytypes.h>

#include <memory>

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
    
    void registerParticleVector         (std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic, int checkpointEvery);
    void registerInteraction            (std::shared_ptr<Interaction> interaction);
    void registerIntegrator             (std::shared_ptr<Integrator> integrator);
    void registerWall                   (std::shared_ptr<Wall> wall, int checkEvery=0);
    void registerBouncer                (std::shared_ptr<Bouncer> bouncer);
    void registerPlugins                (std::shared_ptr<SimulationPlugin> simPlugin, std::shared_ptr<PostprocessPlugin> postPlugin);
    void registerObjectBelongingChecker (std::shared_ptr<ObjectBelongingChecker> checker, ObjectVector* ov);
 
    void setIntegrator  (Integrator* integrator,  ParticleVector* pv);
    void setInteraction (Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2);
    void setBouncer     (Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv);
    void setWallBounce  (Wall* wall, ParticleVector* pv);
    
    
    std::shared_ptr<ParticleVector> makeFrozenWallParticles(Wall* wall, Interaction* interaction);
    
    std::shared_ptr<ParticleVector> applyObjectBelongingChecker(ObjectBelongingChecker* checker,
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
    
    void sayHello();
};
