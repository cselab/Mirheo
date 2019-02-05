#include "interface.h"

Integrator::Integrator(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

Integrator::~Integrator() = default;

void Integrator::setPrerequisites(ParticleVector *pv)
{}
