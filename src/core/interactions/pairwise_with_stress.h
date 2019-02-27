#pragma once
#include "interface.h"
#include "pairwise.h"
#include "pairwise_interactions/stress_wrapper.h"

#include <core/datatypes.h>
#include <map>

template<class PairwiseInteraction>
class InteractionPair_withStress : public Interaction
{
public:
    enum class InteractionType { Regular, Halo };

    InteractionPair_withStress(const YmrState *state, std::string name, float rc, float stressPeriod, PairwiseInteraction pair);
    ~InteractionPair_withStress();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);

    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
private:
    float stressPeriod;
    float lastStressTime{-1e6};

    std::map<ParticleVector*, float> pv2lastStressTime;

    InteractionPair<PairwiseInteraction> interaction;
    InteractionPair<PairwiseStressWrapper<PairwiseInteraction>> interactionWithStress;
};
