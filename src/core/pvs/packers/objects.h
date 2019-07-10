#pragma once

#include "particles.h"

class LocalObjectVector;

struct ObjectPackerHandler : public ParticlePackerHandler
{
    int objSize;
    GenericPackerHandler objects;

    inline __D__ size_t getSizeBytes(int numElements) const
    {
        return ParticlePackerHandler::getSizeBytes(numElements * objSize) +
            objects.getSizeBytes(numElements);
    }
};

class ObjectPacker : public ParticlePacker
{
public:
    ObjectPacker(PackPredicate predicate);
    ~ObjectPacker();
    
    void update(LocalObjectVector *lov, cudaStream_t stream);
    ObjectPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    int objSize;
    GenericPacker objectData;
};
