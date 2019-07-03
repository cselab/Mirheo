#include "objects.h"

#include <core/pvs/object_vector.h>

ObjectPacker::ObjectPacker() = default;
ObjectPacker::~ObjectPacker() = default;

void ObjectPacker::update(LocalObjectVector *lov, PackPredicate& predicate, cudaStream_t stream)
{
    ParticlePacker::update(lov, predicate, stream);
    objectData.updateChannels(lov->dataPerObject, predicate, stream);
    objSize = lov->objSize;
}

ObjectPackerHandler ObjectPacker::handler()
{
    ObjectPackerHandler oh;
    oh.particles = particleData.handler();
    oh.objects   = objectData.handler();
    return oh;
}

size_t ObjectPacker::getSizeBytes(int numElements) const
{
    return ParticlePacker::getSizeBytes(numElements * objSize) +
        objectData.getSizeBytes(numElements);
}
