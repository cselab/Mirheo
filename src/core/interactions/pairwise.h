#pragma once
#include "interface.h"

#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPair : public Interaction
{
public:
    
    InteractionPair(const YmrState *state, std::string name, float rc, PairwiseInteraction pair);
    ~InteractionPair();

    void regular(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo   (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    
    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);    

private:

    PairwiseInteraction defaultPair;
    std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;

    PairwiseInteraction& getPairwiseInteraction(std::string pv1name, std::string pv2name);

    void computeLocal(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);
    void computeHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);
};
