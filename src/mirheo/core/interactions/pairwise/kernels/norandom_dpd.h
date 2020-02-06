#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"

#include <mirheo/core/mirheo_state.h>

#include <random>

namespace mirheo
{

class LocalParticleVector;
class CellList;


class PairwiseNorandomDPD : public PairwiseKernel, public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseNorandomDPD;
    
    PairwiseNorandomDPD(real rc, real a, real gamma, real kBT, real dt, real power) :
        ParticleFetcherWithVelocity(rc),
        a_(a),
        gamma_(gamma),
        sigma_(math::sqrt(2 * gamma_ * kBT / dt)),
        power_(power),
        invrc_(1.0 / rc)
    {}

    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_real3(0.0_r);

        const real invrij = math::rsqrt(rij2);
        const real rij = rij2 * invrij;
        const real argwr = 1.0_r - rij * invrc_;
        const real wr = fastPower(argwr, power_);

        const real3 dr_r = dr * invrij;
        const real3 du = dst.u - src.u;
        const real rdotv = dot(dr_r, du);

        const real myrandnr = ((math::min((int)src.i1, (int)dst.i1) ^ math::max((int)src.i1, (int)dst.i1)) % 13) - 6;

        const real strength = a_ * argwr - (gamma_ * wr * rdotv + sigma_ * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
protected:

    real a_, gamma_, sigma_, power_;
    real invrc_;
};


} // namespace mirheo
