#include "interface.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/macros.h>

Bouncer::Bouncer(const MirState *state, std::string name) :
    MirSimulationObject(state, name)
{}

Bouncer::~Bouncer() = default;

void Bouncer::setPrerequisites(__UNUSED ParticleVector *pv)
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
    exec(pv, cl, ParticleVectorLocality::Local,  stream);
}

void Bouncer::bounceHalo(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    exec(pv, cl, ParticleVectorLocality::Halo, stream);
}

std::vector<std::string> Bouncer::getChannelsToBeSentBack() const
{
    return {};
}
