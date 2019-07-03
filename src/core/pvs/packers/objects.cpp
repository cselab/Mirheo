#include "objects.h"

#include <core/pvs/object_vector.h>

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

size_t ObjectPacker::getSizeBytes(int nObjects, int objSize) const
{
    return ParticlePacker::getSizeBytes(nObjects * objSize) +
        objectData.getSizeBytes(nObjects);
}
