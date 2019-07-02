#include "data_packers.h"

void ParticlePacker::update(LocalParticleVector *lpv, PackPredicate& predicate, cudaStream_t stream)
{
    particleData.updateChannels(lpv->dataPerParticle, predicate, stream);
}

ParticlePackerHandler ParticlePacker::handler()
{
    return {particleData.handler()};
}


void ObjectPacker::update(LocalObjectVector *lov, PackPredicate& predicate, cudaStream_t stream)
{
    ParticlePacker::update(lov, predicate, stream);
    objectData.updateChannels(lov->dataPerObject, predicate, stream);
}

ObjectPackerHandler ObjectPacker::handler()
{
    ObjectPackerHandler oh;
    oh.particles = particleData.handler();
    oh.objects   = objectData.handler();
    return oh;
}


void RodPacker::update(LocalRodVector *lrv, PackPredicate& predicate, cudaStream_t stream)
{
    ObjectPacker::update(lrv, predicate, stream);
    bisegmentData.updateChannels(lrv->dataPerBisegment, predicate, stream);
}

RodPackerHandler RodPacker::handler()
{
    RodPackerHandler rh;
    rh.particles  = particleData.handler();
    rh.objects    = objectData.handler();
    rh.bisegments = bisegmentData.handler();
    return rh;
}
