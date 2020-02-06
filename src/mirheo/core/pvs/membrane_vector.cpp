#include "membrane_vector.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

MembraneVector::MembraneVector(const MirState *state, const std::string& name, real mass, std::shared_ptr<MembraneMesh> mptr, int nObjects) :
    ObjectVector( state, name, mass, mptr->getNvertices(),
                  std::make_unique<LocalObjectVector>(this, mptr->getNvertices(), nObjects),
                  std::make_unique<LocalObjectVector>(this, mptr->getNvertices(), 0) )
{
    mesh = std::move(mptr);
}

MembraneVector::~MembraneVector() = default;

ConfigDictionary MembraneVector::writeSnapshot(Dumper& dumper)
{
    ConfigDictionary dict = ObjectVector::writeSnapshot(dumper);
    dict.insert_or_assign("__type", dumper("MembraneVector"));
    return dict;
}

} // namespace mirheo
