#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/pvs/membrane_vector.h>

class FilterKeepByTypeId
{
public:
    FilterKeepByTypeId(int whiteListTypeId) :
        whiteListTypeId(whiteListTypeId)
    {}

    void setPrerequisites(MembraneVector *mv) const
    {
        mv->requireDataPerObject<int>(ChannelNames::membraneTypeId,
                                      DataManager::PersistenceMode::Active,
                                      DataManager::ShiftMode::None);
    }

    void setup(MembraneVector *mv)
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
    int whiteListTypeId {-1};
    const int *typeIds {nullptr};
};
