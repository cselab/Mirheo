#pragma once

#include "object_vector.h"

#include <mirheo/core/mesh/membrane.h>

namespace mirheo
{

class MembraneVector: public ObjectVector
{
public:
    MembraneVector(const MirState *state, const std::string& name, real mass, std::shared_ptr<MembraneMesh> mptr, int nObjects = 0);

    /** \brief Load a membrane vector form a snapshot.
        \param [in] state The simulation state.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The membrane vector parameters.
     */
    MembraneVector(const MirState *state, Loader& loader, const ConfigObject& config);

    ~MembraneVector();

    /** \brief Dump the membrane vector state, create a \c ConfigObject with its metadata and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c MembraneVector.
      */
    void saveSnapshotAndRegister(Saver& saver) override;
};

} // namespace mirheo
