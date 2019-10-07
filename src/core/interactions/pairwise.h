#pragma once

#include "interface.h"
#include "pairwise/kernels/parameters.h"

class PairwiseInteraction : public Interaction
{
public:
    
    PairwiseInteraction(const MirState *state, const std::string& name, float rc, VarPairwiseParams varParams);
    ~PairwiseInteraction();

private:
    VarPairwiseParams varParams;
};
