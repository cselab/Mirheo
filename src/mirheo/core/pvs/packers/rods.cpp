#include "rods.h"

#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

RodPacker::RodPacker(PackPredicate predicate) :
    ObjectPacker(predicate)
{}

RodPacker::~RodPacker() = default;

void RodPacker::update(LocalParticleVector *lpv, cudaStream_t stream)
{
    ObjectPacker::update(lpv, stream);

    auto lrv = dynamic_cast<LocalRodVector*>(lpv);
    if (lrv == nullptr) die("Must pass local rod vector to rod packer update");

    bisegmentData_.updateChannels(lrv->dataPerBisegment, predicate_, stream);
    nBisegments_ = lrv->getNumSegmentsPerRod() - 1;
}

RodPackerHandler RodPacker::handler()
{
    RodPackerHandler rh;
    rh.particles   = particleData_.handler();
    rh.objSize     = objSize_;
    rh.objects     = objectData_.handler();
    rh.nBisegments = nBisegments_;
    rh.bisegments  = bisegmentData_.handler();
    return rh;
}

size_t RodPacker::getSizeBytes(int numElements) const
{
    return ObjectPacker::getSizeBytes(numElements) +
        bisegmentData_.getSizeBytes(nBisegments_ * numElements);
}

} // namespace mirheo
