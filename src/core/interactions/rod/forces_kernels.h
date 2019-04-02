#pragma once

#include "real.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

namespace RodForcesKernels
{

struct GPU_RodParameters
{
};


__global__ void computeRodForces(RVview view, GPU_RodParameters parameters)
{
    // TODO
}

} // namespace RodForcesKernels
