#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/density.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>
#include <core/pvs/views/pv.h>

class CellList;
class LocalParticleVector;

#ifndef __NVCC__
static float fastPower(float x, float a)
{
    return pow(x, a);
}
#else
#include <core/utils/cuda_common.h>
#endif


class Pairwise_density
{
public:

    using ViewType = PVviewWithDensities;
    
    Pairwise_density(float rc) :
        rc(rc)
    {
        rc2 = rc*rc;
        invrc = 1.0 / rc;
        fact = 15.0 / (2 * M_PI * rc2 * rc);
    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {}

    __D__ inline float operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        float3 dr = dst.r - src.r;
        float rij2 = dot(dr, dr);
        if (rij2 > rc2) return 0.0f;

        float rij = sqrtf(rij2);
        float argwr = 1.0f - rij * invrc;

        return fact * argwr * argwr;
    }

    __D__ inline DensityAccumulator getZeroedAccumulator() const {return DensityAccumulator();}

protected:

    float rc;
    float invrc, rc2, fact;
};
