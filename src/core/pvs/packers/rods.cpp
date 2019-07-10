#include "rods.h"

#include <core/pvs/rod_vector.h>

RodPacker::RodPacker(PackPredicate predicate) :
    ObjectPacker(predicate)
{}

RodPacker::~RodPacker() = default;

void RodPacker::update(LocalRodVector *lrv, cudaStream_t stream)
{
    ObjectPacker::update(lrv, stream);
    bisegmentData.updateChannels(lrv->dataPerBisegment, predicate, stream);
    nBisegments = lrv->getNumSegmentsPerRod();
}

RodPackerHandler RodPacker::handler()
{
    RodPackerHandler rh;
    rh.particles   = particleData.handler();
    rh.objects     = objectData.handler();
    rh.nBisegments = nBisegments;
    rh.bisegments  = bisegmentData.handler();
    return rh;
}

size_t RodPacker::getSizeBytes(int numElements) const
{
    return ObjectPacker::getSizeBytes(numElements) +
        bisegmentData.getSizeBytes(nBisegments * numElements);
}
