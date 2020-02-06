#include "objects.h"

#include <mirheo/core/pvs/object_vector.h>

namespace mirheo
{

ObjectPacker::ObjectPacker(PackPredicate predicate) :
    ParticlePacker(predicate)
{}

ObjectPacker::~ObjectPacker() = default;

void ObjectPacker::update(LocalParticleVector *lpv, cudaStream_t stream)
{
    ParticlePacker::update(lpv, stream);

    auto lov = dynamic_cast<LocalObjectVector*>(lpv);
    if (lov == nullptr) die("Must pass local object vector to object packer update");
    
    objectData_.updateChannels(lov->dataPerObject, predicate_, stream);
    objSize_ = lov->getObjectSize();
}

ObjectPackerHandler ObjectPacker::handler()
{
    ObjectPackerHandler oh;
    oh.particles = particleData_.handler();
    oh.objSize   = objSize_;
    oh.objects   = objectData_.handler();
    return oh;
}

size_t ObjectPacker::getSizeBytes(int numElements) const
{
    return ParticlePacker::getSizeBytes(numElements * objSize_) +
        objectData_.getSizeBytes(numElements);
}

} // namespace mirheo
