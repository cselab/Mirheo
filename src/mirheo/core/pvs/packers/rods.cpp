#include "rods.h"

#include <core/pvs/rod_vector.h>

RodPacker::RodPacker(PackPredicate predicate) :
    ObjectPacker(predicate)
{}

RodPacker::~RodPacker() = default;

void RodPacker::update(LocalParticleVector *lpv, cudaStream_t stream)
{
    ObjectPacker::update(lpv, stream);

    auto lrv = dynamic_cast<LocalRodVector*>(lpv);
    if (lrv == nullptr) die("Must pass local rod vector to rod packer update");

    bisegmentData.updateChannels(lrv->dataPerBisegment, predicate, stream);
    nBisegments = lrv->getNumSegmentsPerRod() - 1;
}

RodPackerHandler RodPacker::handler()
{
    RodPackerHandler rh;
    rh.particles   = particleData.handler();
    rh.objSize     = objSize;
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
