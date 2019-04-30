#include "interface.h"

#include <core/pvs/particle_vector.h>

Integrator::Integrator(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

Integrator::~Integrator() = default;

void Integrator::setPrerequisites(ParticleVector *pv)
{}

void Integrator::invalidatePV(ParticleVector *pv)
{
    pv->haloValid   = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}
