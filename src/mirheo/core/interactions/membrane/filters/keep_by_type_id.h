#pragma once

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class FilterKeepByTypeId
{
public:
    FilterKeepByTypeId(int whiteListTypeId) :
        whiteListTypeId_(whiteListTypeId)
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
        typeIds_ = typeIdsBuff->devPtr();
    }

    inline __D__ bool inWhiteList(long membraneId) const
    {
        const auto typeId = typeIds_[membraneId];
        return typeId == whiteListTypeId_;
    }

private:
    int whiteListTypeId_ {-1};
    const int *typeIds_ {nullptr};

    friend MemberVars<FilterKeepByTypeId>;
};

MIRHEO_MEMBER_VARS_1(FilterKeepByTypeId, whiteListTypeId_);

} // namespace mirheo
