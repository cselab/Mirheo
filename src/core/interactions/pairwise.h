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
    enum class InteractionType { Regular, Halo };

    void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);

    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

    InteractionPair(std::string name, float rc, PairwiseInteraction pair) :
        Interaction(name, rc), defaultPair(pair)
    { }

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);

    ~InteractionPair() = default;

private:
    PairwiseInteraction defaultPair;
    std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;
};
