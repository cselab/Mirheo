#pragma once

#include "interface.h"

class ObjectVector;
class LocalObjectVector;
class ExchangeHelper;
struct BufferInfos;

class ObjectsPacker : public Packer
{
public:
    ObjectsPacker(ObjectVector *ov, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) const override;

    void packToBuffer(const LocalObjectVector *lov, ExchangeHelper *helper, cudaStream_t stream);
    void unpackFromBuffer(LocalObjectVector *lov, const BufferInfos *helper, int oldObjSize, cudaStream_t stream);

protected:
    ObjectVector *ov;
};
