#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/force.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/helper_math.h>

#include <random>

class LocalParticleVector;
class CellList;


class Pairwise_Norandom_DPD
{
public:
    Pairwise_Norandom_DPD(float rc, float a, float gamma, float kbT, float dt, float power) :
        rc(rc), a(a), gamma(gamma), power(power)
    {
        sigma = sqrt(2 * gamma * kbT / dt);
        rc2 = rc*rc;
        invrc = 1.0 / rc;
    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {    }

    __D__ inline float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float wr = fastPower(argwr, power);

        const float3 dr_r = dr * invrij;
        const float3 du = dst.u - src.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = ((min(src.i1, dst.i1) ^ max(src.i1, dst.i1)) % 13) - 6;

        const float strength = a * argwr - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}
    
protected:

    float a, gamma, sigma, power, rc;
    float invrc, rc2;
};
