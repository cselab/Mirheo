#pragma once

#include <core/utils/cpu_gpu_defines.h>

class FilterKeepAll
{
public:
    FilterKeepAll() = default;

    inline __D__ bool inWhiteList(long membraneId) const
    {
        return true;
    }
};
