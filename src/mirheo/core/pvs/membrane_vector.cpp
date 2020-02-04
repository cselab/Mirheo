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

Config MembraneVector::writeSnapshot(Dumper &dumper) const
{
    auto config = ObjectVector::writeSnapshot(dumper);
    Config::Dictionary &dict = config.getDict();
    dict.at("__type") = Config{"MembraneVector"};
    return config;
}

} // namespace mirheo
