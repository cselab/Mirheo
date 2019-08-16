#pragma once

#include "interface.h"
#include <memory>

class InteractionLJ : public Interaction
{
public:
    enum class AwareMode {None, Object, Rod};
    
    InteractionLJ(const MirState *state, std::string name, float rc, float epsilon, float sigma, float maxForce, AwareMode awareness, int minSegmentsDist=0);

    ~InteractionLJ();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    virtual void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                 float epsilon, float sigma, float maxForce);

protected:
    InteractionLJ(const MirState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                  AwareMode awareness, int minSegmentsDist, bool allocate);
    
    AwareMode awareness;
    int minSegmentsDist;
};

