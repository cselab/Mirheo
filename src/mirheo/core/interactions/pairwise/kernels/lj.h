// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

namespace mirheo
{

/// Compute Lennard-Jones forces on the device.
class PairwiseLJ : public PairwiseKernel, public ParticleFetcher
{
public:
    // TODO: (Ivica) HandlerType should not include the whole PairwiseLJ, but
    // only the Fetcher, since the PairwiseKernel includes some virtual
    // functions. Check the DPD kernel for a proper implementation.
    using ViewType     = PVview;     ///< Compatible view type
    using ParticleType = Particle;   ///< Compatible particle type
    using HandlerType  = PairwiseLJ; ///< Corresponding handler
    using ParamsType   = LJParams;   ///< Corresponding parameters type

    /// Constructor
    PairwiseLJ(real rc, real epsilon, real sigma) :
        ParticleFetcher(rc),
        sigma2_(sigma * sigma),
        epsx24_sigma2_(24.0_r * epsilon / (sigma * sigma))
    {}

    /// Generic constructor
    PairwiseLJ(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        PairwiseLJ{rc, p.epsilon, p.sigma}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, int /*dstId*/,
                                  ParticleType src, int /*srcId*/) const
    {
        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);
        if (dr2 > rc2_)
            return make_real3(0.0_r);

        const real rs2 = sigma2_ / dr2;
        const real rs4 = rs2 * rs2;
        const real rs8 = rs4 * rs4;
        const real rs14 = rs8 * (rs4 * rs2);
        const real IfI = epsx24_sigma2_ * (2*rs14 - rs8);

        return IfI * dr;
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

private:
    real sigma2_;
    real epsx24_sigma2_;
};

} // namespace mirheo
