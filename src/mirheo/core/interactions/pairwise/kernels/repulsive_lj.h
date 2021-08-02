// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "awareness.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

/** \brief Compute repulsive Lennard-Jones forces on the device
    \tparam Awareness A functor that describes which particles pairs interact
 */
template <class Awareness>
class PairwiseRepulsiveLJ : public PairwiseKernel, public ParticleFetcher
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using ViewType     = PVview;              ///< Compatible view type
    using ParticleType = Particle;            ///< Compatible particle type
    using HandlerType  = PairwiseRepulsiveLJ<Awareness>; ///< Corresponding handler
    using ParamsType   = RepulsiveLJParams;   ///< Corresponding parameters type
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseRepulsiveLJ(real rc, real epsilon, real sigma, real maxForce, Awareness awareness) :
        ParticleFetcher(rc),
        sigma2_(sigma*sigma),
        maxForce_(maxForce),
        epsx24_sigma2_(24.0_r * epsilon / (sigma * sigma)),
        awareness_(awareness)
    {
        constexpr real sigmaFactor = 1.1224620483_r; // 2^(1/6)
        const real rm = sigmaFactor * sigma; // F(rm) = 0

        if (rm > rc)
        {
            const real maxSigma = rc / sigmaFactor;
            die("RepulsiveLJ: rm = %g > rc = %g; sigma must be lower than %g or rc must be larger than %g",
                rm, rc, maxSigma, rm);
        }
    }

    /// Generic constructor
    PairwiseRepulsiveLJ(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        PairwiseRepulsiveLJ{rc,
                            p.epsilon,
                            p.sigma,
                            p.maxForce,
                            mpark::get<typename Awareness::ParamsType>(p.varAwarenessParams)}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        constexpr real tolerance = 1e-6_r;
        if (!awareness_.interact(src.i1, dst.i1))
            return make_real3(0.0_r);

        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);

        if (dr2 > rc2_ || dr2 < tolerance)
            return make_real3(0.0_r);

        const real rs2 = sigma2_ / dr2;
        const real rs4 = rs2*rs2;
        const real rs8 = rs4*rs4;
        const real rs14 = rs8*(rs4*rs2);

        const real IfI = epsx24_sigma2_ * (2*rs14 - rs8);

        return dr * math::min(math::max(IfI, 0.0_r), maxForce_);
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2,
               __UNUSED CellList *cl1, __UNUSED CellList *cl2, __UNUSED const MirState *state) override
    {
        awareness_.setup(lpv1, lpv2);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return constructTypeName<Awareness>("PairwiseRepulsiveLJ");
    }

private:
    real sigma2_, maxForce_;
    real epsx24_sigma2_;

    Awareness awareness_;
};

} // namespace mirheo
