// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_vector.h"

namespace mirheo
{

/** \brief Represent a set of chains with the same length
 */
class ChainVector: public ObjectVector
{
public:
    /** Construct a ChainVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] chainLength Number of particles per chain
        \param [in] nObjects Number of objects
    */
    ChainVector(const MirState *state, const std::string& name, real mass, int chainLength, int nObjects = 0);

    ~ChainVector();
};

} // namespace mirheo
