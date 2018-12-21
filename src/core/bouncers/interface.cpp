#include "interface.h"

Bouncer::Bouncer(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

virtual Bouncer::~Bouncer() = default;

void Bouncer::setPrerequisites(ParticleVector* pv)
{}

void Bouncer::bounceLocal(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream)
{
    exec(pv, cl, dt, true,  stream);
}

void Bouncer::bounceHalo(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream)
{
    exec(pv, cl, dt, false, stream);
}
