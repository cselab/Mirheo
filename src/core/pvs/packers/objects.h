#pragma once

#include "particles.h"

class LocalObjectVector;

struct ObjectPackerHandler : public ParticlePackerHandler
{
    GenericPackerHandler objects;
};

class ObjectPacker : public ParticlePacker
{
public:
    void update(LocalObjectVector *lov, PackPredicate& predicate, cudaStream_t stream);
    ObjectPackerHandler handler();
    size_t getSizeBytes(int nObjects, int objSize) const;

protected:
    GenericPacker objectData;
};
