#include "objects.h"

#include <mirheo/core/pvs/object_vector.h>

ObjectPacker::ObjectPacker(PackPredicate predicate) :
    ParticlePacker(predicate)
{}

ObjectPacker::~ObjectPacker() = default;

void ObjectPacker::update(LocalParticleVector *lpv, cudaStream_t stream)
{
    ParticlePacker::update(lpv, stream);

    auto lov = dynamic_cast<LocalObjectVector*>(lpv);
    if (lov == nullptr) die("Must pass local object vector to object packer update");
    
    objectData.updateChannels(lov->dataPerObject, predicate, stream);
    objSize = lov->objSize;
}

ObjectPackerHandler ObjectPacker::handler()
{
    ObjectPackerHandler oh;
    oh.particles = particleData.handler();
    oh.objSize   = objSize;
    oh.objects   = objectData.handler();
    return oh;
}

size_t ObjectPacker::getSizeBytes(int numElements) const
{
    return ParticlePacker::getSizeBytes(numElements * objSize) +
        objectData.getSizeBytes(numElements);
}
