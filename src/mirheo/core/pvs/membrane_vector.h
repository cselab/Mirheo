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

    /** \brief Load a membrane vector form a snapshot.
        \param [in] state The simulation state.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The membrane vector parameters.
     */
    MembraneVector(const MirState *state, Loader& loader, const ConfigObject& config);

    ~MembraneVector();

    /** \brief Dump the membrane vector state, create a ConfigObject with its metadata and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly MembraneVector.
      */
    void saveSnapshotAndRegister(Saver& saver) override;
};

} // namespace mirheo
