// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

class MembraneVector;

/// Filter that keeps all the membranes
class FilterKeepAll
{
public:
    /// set required properties to \p mv
    void setPrerequisites(__UNUSED MembraneVector *mv) const {}
    /// Set internal state of the object
    void setup           (__UNUSED MembraneVector *mv)       {}

    /** \brief States if the given membrane must be kept or not
        \param [in] membraneId The index of the membrane to keep or not
        \return \c true if the membrane should be kept, \c false otherwise.
     */
    inline __D__ bool inWhiteList(__UNUSED long membraneId) const
    {
        return true;
    }
};

} // namespace mirheo
