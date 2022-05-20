// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "chain_vector.h"

namespace mirheo
{

ChainVector::ChainVector(const MirState *state, const std::string& name, real mass, int chainLength, int nObjects) :
    ObjectVector( state, name, mass, chainLength,
                  std::make_unique<LocalObjectVector>(this, chainLength, nObjects),
                  std::make_unique<LocalObjectVector>(this, chainLength, 0) )
{
    if (chainLength < 2)
        die("A chain must have at least 2 particles.");
}

ChainVector::~ChainVector() = default;

} // namespace mirheo
