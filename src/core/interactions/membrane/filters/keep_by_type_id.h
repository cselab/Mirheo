#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/pvs/membrane_vector.h>

class FilterKeepByTypeId
{
public:
    FilterKeepByTypeId(int whiteListTypeId, MembraneVector *mv) :
        whiteListTypeId(whiteListTypeId)
    {
        LocalObjectVector *lmv = mv->local();
        auto typeIdsBuff = lmv->dataPerObject.getData<int>(ChannelNames::membraneTypeId);
        typeIds = typeIdsBuff->devPtr();
    }

    inline __D__ bool inWhiteList(long membraneId) const
    {
        const auto typeId = typeIds[membraneId];
        return typeId == whiteListTypeId;
    }

private:
    int whiteListTypeId;
    const int *typeIds {nullptr};
};
