#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/macros.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class MembraneVector;

class FilterKeepAll
{
public:
    void setPrerequisites(__UNUSED MembraneVector *mv) const {}
    void setup           (__UNUSED MembraneVector *mv)       {}
    
    inline __D__ bool inWhiteList(__UNUSED long membraneId) const
    {
        return true;
    }
};

MIRHEO_MEMBER_VARS(0, FilterKeepAll);

} // namespace mirheo
