#include "interface.h"

#include <core/pvs/particle_vector.h>

Integrator::Integrator(const MirState *state, std::string name) :
    MirSimulationObject(state, name)
{}

Integrator::~Integrator() = default;

void Integrator::setPrerequisites(__UNUSED ParticleVector *pv)
{}

void Integrator::invalidatePV(ParticleVector *pv)
{
    pv->haloValid   = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}
