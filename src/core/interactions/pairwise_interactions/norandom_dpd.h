#pragma once

#include "fetchers.h"

#include <core/interactions/accumulators/force.h>
#include <core/ymero_state.h>

#include <random>

class LocalParticleVector;
class CellList;


class PairwiseNorandomDPDHandler : public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    
    PairwiseNorandomDPDHandler(float rc, float a, float gamma, float kbT, float dt, float power) :
        ParticleFetcherWithVelocity(rc),
        a(a), gamma(gamma), power(power)
    {
        sigma = sqrt(2 * gamma * kbT / dt);
        invrc = 1.0 / rc;
    }

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
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

    float a, gamma, sigma, power;
    float invrc;
};

class PairwiseNorandomDPD : public PairwiseNorandomDPDHandler
{
public:

    using HandlerType = PairwiseNorandomDPDHandler;
    
    PairwiseNorandomDPD(float rc, float a, float gamma, float kbT, float dt, float power) :
        PairwiseNorandomDPDHandler(rc, a, gamma, kbT, dt, power)
    {}
    
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, const YmrState *state)
    {}
};
