// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

/** Keep membranes that have a given \c typeId.
    The \c typeId of each membrane is stored in the object channel \c ChannelNames::membraneTypeId.
 */
class FilterKeepByTypeId
{
public:
    /** \brief Construct FilterKeepByTypeId that wil keep only the membranes with type id \p whiteListTypeId
        \param [in] whiteListTypeId The type id of the membranes to keep
     */
    FilterKeepByTypeId(int whiteListTypeId) :
        whiteListTypeId_(whiteListTypeId)
    {}

    /** \brief set required properties to \p mv
        \param [out] mv The MembraneVector taht will be used
    */
    void setPrerequisites(MembraneVector *mv) const
    {
        mv->requireDataPerObject<int>(channel_names::membraneTypeId,
                                      DataManager::PersistenceMode::Active,
                                      DataManager::ShiftMode::None);
    }

    /** \brief Set internal state of the object.
        \param [in] mv The MembraneVector tahat will be used

        This must be called after every change of \p mv DataManager
     */
    void setup(MembraneVector *mv)
    {
        LocalObjectVector *lmv = mv->local();
        auto typeIdsBuff = lmv->dataPerObject.getData<int>(channel_names::membraneTypeId);
        typeIds_ = typeIdsBuff->devPtr();
    }

    /** \brief States if the given membrane must be kept or not
        \param [in] membraneId The index of the membrane to keep or not
        \return \c true if the membrane should be kept, \c false otherwise.
     */
    inline __D__ bool inWhiteList(long membraneId) const
    {
        const auto typeId = typeIds_[membraneId];
        return typeId == whiteListTypeId_;
    }

private:
    int whiteListTypeId_; ///< The type id of the membranes that will be kept
    const int *typeIds_ {nullptr}; ///< Points to the MembraneVector object channel containing type ids
};

} // namespace mirheo
