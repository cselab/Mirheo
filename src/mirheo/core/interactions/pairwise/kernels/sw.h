// Copyright 2020 ETH Zurich. All Rights Reserved.

// Code: Noah Baumann Bachelor Thesis
// SW: Stillinger-Weber potiential
// 2Body SW potential

// Paper: "Water Modeled As an Intermediate Element between Carbon and Silicon 2009"
// https://pubs.acs.org/doi/abs/10.1021/jp805227c
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/utils/cuda_common.h>      //math::exp(), math:sqrt()

namespace mirheo
{

/// Compute Stillinger-Weber forces on the device.
class PairwiseSW : public PairwiseKernel, public ParticleFetcher
{
public:
    using ViewType     = PVview;     ///< Compatible view type
    using ParticleType = Particle;   ///< Compatible particle type
    using HandlerType  = PairwiseSW; ///< Corresponding handler
    using ParamsType   = SW2Params;  ///< Corresponding parameters type

    /// Constructor
    PairwiseSW(real rc, real epsilon, real sigma, real A, real B) :
        ParticleFetcher(rc),
        epsilon_(epsilon),
        sigma_(sigma),
        A_(A),
        B_(B)
    {}

    /// Generic constructor
    PairwiseSW(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        PairwiseSW{rc, p.epsilon, p.sigma, p.A, p.B}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, int /*dstId*/,
                                  ParticleType src, int /*srcId*/) const
    {
        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);
        if (dr2 > rc2_)
            return make_real3(0.0_r);

        const real rs2 = (sigma_*sigma_) / dr2;
        const real B_rs4 = B_*rs2 * rs2;
        const real dr_ = math::sqrt(dr2);
        const real r_rc = dr_ - rc_;
        const real exp = math::exp(sigma_ / r_rc);
        const real A_eps_exp = A_*epsilon_*exp;
        const real phi = (sigma_*(B_rs4 - 1.0_r))/(r_rc*r_rc*dr_) + (4.0_r*B_rs4)/dr2;

        return A_eps_exp * phi * dr;
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

private:
    real epsilon_;
    real sigma_;
    real A_;
    real B_;
};

} // namespace mirheo
