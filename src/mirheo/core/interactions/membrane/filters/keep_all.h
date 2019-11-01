#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>

class MembraneVector;

class FilterKeepAll
{
public:
    FilterKeepAll() = default;

    void setPrerequisites(__UNUSED MembraneVector *mv) const {}
    void setup           (__UNUSED MembraneVector *mv)       {}
    
    inline __D__ bool inWhiteList(__UNUSED long membraneId) const
    {
        return true;
    }
};
