// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "base_pairwise.h"

namespace mirheo
{

BasePairwiseInteraction::BasePairwiseInteraction(const MirState *state, const std::string& name, real rc) :
    Interaction(state, name),
    rc_(rc)
{}

BasePairwiseInteraction::~BasePairwiseInteraction() = default;

real BasePairwiseInteraction::getCutoffRadius() const
{
    return rc_;
}

} // namespace mirheo
