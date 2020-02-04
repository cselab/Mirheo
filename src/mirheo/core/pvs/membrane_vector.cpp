#include "membrane_vector.h"

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

} // namespace mirheo
