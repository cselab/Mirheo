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

ConfigDictionary Integrator::_saveSnapshot(Dumper& dumper, const std::string& typeName)
{
    return MirSimulationObject::_saveSnapshot(dumper, "Integrator", typeName);
}

} // namespace mirheo
