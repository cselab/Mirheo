#pragma once

#include "objects.h"

class LocalRodVector;

struct RodPackerHandler : public ObjectPackerHandler
{
    GenericPackerHandler bisegments;
};


class RodPacker : public ObjectPacker
{
public:

    RodPacker(PackPredicate predicate);
    ~RodPacker();
    
    void update(LocalRodVector *lrv, cudaStream_t stream);
    RodPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    GenericPacker bisegmentData;
    int nBisegments;
};
