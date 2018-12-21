#include "interface.h"

Integrator::Integrator(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name),
    dt(state->dt)
{}

Integrator::~Integrator() = default;

void Integrator::setPrerequisites(ParticleVector *pv)
{}
