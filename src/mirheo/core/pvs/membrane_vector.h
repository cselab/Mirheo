// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_vector.h"

#include <mirheo/core/mesh/membrane.h>

namespace mirheo
{

/** \brief Represent a set of membranes

    Each membrane is composed of the same connectivity (stored in mesh) and number of vertices.
    The particles data correspond to the vertices of the membranes.
 */
class MembraneVector: public ObjectVector
{
public:
    /** Construct a MembraneVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] mptr Triangle mesh which stores the connectivity of a single membrane
        \param [in] nObjects Number of objects
    */
    MembraneVector(const MirState *state, const std::string& name, real mass, std::shared_ptr<MembraneMesh> mptr, int nObjects = 0);

    ~MembraneVector();
};

} // namespace mirheo
