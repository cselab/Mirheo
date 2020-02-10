#pragma once

#include "object_vector.h"

#include <mirheo/core/mesh/membrane.h>

namespace mirheo
{

class MembraneVector: public ObjectVector
{
public:
    MembraneVector(const MirState *state, const std::string& name, real mass, std::shared_ptr<MembraneMesh> mptr, int nObjects = 0);
    MembraneVector(const MirState *state, Undumper&, const ConfigDictionary&);
    ~MembraneVector();

    void saveSnapshotAndRegister(Dumper& dumper) override;
};

} // namespace mirheo
