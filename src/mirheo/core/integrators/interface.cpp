#include "interface.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

Integrator::Integrator(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

Integrator::~Integrator() = default;

void Integrator::setPrerequisites(__UNUSED ParticleVector *pv)
{}

void Integrator::invalidatePV_(ParticleVector *pv)
{
    pv->haloValid   = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}

ConfigObject Integrator::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    return MirSimulationObject::_saveSnapshot(saver, "Integrator", typeName);
}

} // namespace mirheo
