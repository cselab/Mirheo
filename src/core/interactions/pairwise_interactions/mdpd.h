#pragma once

#include "fetchers.h"

#include <core/interactions/accumulators/force.h>
#include <core/interactions/utils/step_random_gen.h>
#include <core/ymero_state.h>

#include <random>

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

class PairwiseMDPDHandler : public ParticleFetcherWithVelocityAndDensity
{
public:

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensity;
    
    PairwiseMDPDHandler(float rc, float rd, float a, float b, float gamma, float kbT, float dt, float power) :
        ParticleFetcherWithVelocityAndDensity(rc),
        rd(rd), a(a), b(b), gamma(gamma), power(power)
    {
        sigma = sqrt(2 * gamma * kbT / dt);
        invrc = 1.0 / rc;
        invrd = 1.0 / rd;
    }

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const float3 dr = dst.p.r - src.p.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float argwd = max(1.0f - rij * invrd, 0.f);

        const float wr = fastPower(argwr, power);

        const float3 dr_r = dr * invrij;
        const float3 du = dst.p.u - src.p.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = Logistic::mean0var1(seed, min(src.p.i1, dst.p.i1), max(src.p.i1, dst.p.i1));

        const float strength = a * argwr + b * argwd * (src.d + dst.d) - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    float a, b, gamma, sigma, power, rd;
    float invrc, invrd;
    float seed;
};

class PairwiseMDPD : public PairwiseMDPDHandler
{
public:

    using HandlerType = PairwiseMDPDHandler;
    
    PairwiseMDPD(float rc, float rd, float a, float b, float gamma, float kbT, float dt, float power, long seed = 42424242) :
        PairwiseMDPDHandler(rc, rd, a, b, gamma, kbT, dt, power),
        stepGen(seed)
    {}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state)
    {
        seed = stepGen.generate(state);
    }

protected:

    StepRandomGen stepGen;
};
