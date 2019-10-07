#include "pairwise.h"

PairwiseInteraction::PairwiseInteraction(const MirState *state, const std::string& name, float rc, VarPairwiseParams varParams) :
    Interaction(state, name, rc),
    varParams(varParams)
{}

PairwiseInteraction::~PairwiseInteraction() = default;


