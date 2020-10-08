// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/mirheo_state.h>

#include <random>

namespace mirheo
{

class LocalParticleVector;
class CellList;

/// a GPU compatible functor that computes DPD interactions without fluctuations.
/// Used in unit tests
class PairwiseNorandomDPD : public PairwiseKernel, public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;   ///< compatible view type
    using ParticleType = Particle; ///< compatible particle type
    using HandlerType  = PairwiseNorandomDPD;  ///< handler type corresponding to this object
    using ParamsType   = NoRandomDPDParams; ///< parameters that are used to create this object

    /// constructor
    PairwiseNorandomDPD(real rc, real a, real gamma, real kBT, real power) :
        ParticleFetcherWithVelocity(rc),
        a_(a),
        gamma_(gamma),
        kBT_(kBT),
        power_(power),
        invrc_(1.0 / rc)
    {}

    /// Generic constructor
    PairwiseNorandomDPD(real rc, const ParamsType& p, long seed=42424242) :
        PairwiseNorandomDPD(rc, p.a, p.gamma, p.kBT, p.power)
    {}

    /// evaluate the force
    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2_) return make_real3(0.0_r);

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

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    void setup(__UNUSED LocalParticleVector *lpv1,
               __UNUSED LocalParticleVector *lpv2,
               __UNUSED CellList *cl1,
               __UNUSED CellList *cl2,
               const MirState *state) override
    {
        sigma_ = math::sqrt(2 * gamma_ * kBT_ / state->getDt());
    }


    /// \return type name string
    static std::string getTypeName()
    {
        return "PairwiseNorandomDPD";
    }

protected:
    real a_; ///< conservative force magnitude
    real gamma_; ///< viscous force coefficient
    real sigma_{NAN}; ///< random force coefficient, depends on dt
    real kBT_; ///< temperature
    real power_; ///< viscous kernel envelope power
    real invrc_; ///< 1 / rc
};


} // namespace mirheo
