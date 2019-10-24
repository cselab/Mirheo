#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"

#include <core/mirheo_state.h>

#include <random>

class LocalParticleVector;
class CellList;


class PairwiseNorandomDPD : public PairwiseKernel, public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseNorandomDPD;
    
    PairwiseNorandomDPD(float rc, float a, float gamma, float kBT, float dt, float power) :
        ParticleFetcherWithVelocity(rc),
        a(a),
        gamma(gamma),
        sigma(math::sqrt(2 * gamma * kBT / dt)),
        power(power),
        invrc(1.0 / rc)
    {}

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = math::rsqrt(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float wr = fastPower(argwr, power);

        const float3 dr_r = dr * invrij;
        const float3 du = dst.u - src.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = ((math::min(src.i1, dst.i1) ^ math::max(src.i1, dst.i1)) % 13) - 6;

        const float strength = a * argwr - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
protected:

    float a, gamma, sigma, power;
    float invrc;
};

