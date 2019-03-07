#include "interface.h"

Bouncer::Bouncer(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

Bouncer::~Bouncer() = default;

void Bouncer::setPrerequisites(ParticleVector *pv)
{}

void Bouncer::setup(ObjectVector *ov)
{
    this->ov = ov;
}

ObjectVector* Bouncer::getObjectVector()
{
    return ov;
}

void Bouncer::bounceLocal(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    exec(pv, cl, true,  stream);
}

void Bouncer::bounceHalo(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    exec(pv, cl, false, stream);
}
