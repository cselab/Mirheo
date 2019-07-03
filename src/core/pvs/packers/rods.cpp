#include "rods.h"

#include <core/pvs/rod_vector.h>

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

size_t RodPacker::getSizeBytes(int nObjects, int objSize) const
{
    int nBiSegmentsPerObj = (objSize - 1) / 5 - 1;
    int nBisegments = nBiSegmentsPerObj * nObjects;
    return ObjectPacker::getSizeBytes(nObjects, objSize) +
        bisegmentData.getSizeBytes(nBisegments);
}
