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

    RodPacker();
    ~RodPacker();
    
    void update(LocalRodVector *lrv, PackPredicate& predicate, cudaStream_t stream);
    RodPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    GenericPacker bisegmentData;
    int nBisegments;
};
