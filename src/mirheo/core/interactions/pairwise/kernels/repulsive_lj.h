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




/** \brief A GPU compatible functor to compute repulsive LJ interactions
    \tparam Awareness A functor that describes which particles pairs interact
 */
template <class Awareness>
class PairwiseRepulsiveLJHandler : public ParticleFetcher
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using ViewType     = PVview;              ///< Compatible view type
    using ParticleType = Particle;            ///< Compatible particle type
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseRepulsiveLJHandler(real rc, real epsilon, real sigma, real maxForce, Awareness awareness) :
        ParticleFetcher(rc),
        sigma2_(sigma*sigma),
        maxForce_(maxForce),
        epsx24_sigma2_(24.0_r * epsilon / (sigma * sigma)),
        awareness_(awareness)
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

private:
    real sigma2_;
    real maxForce_;
    real epsx24_sigma2_;
    Awareness awareness_;
};


/// Kernel for repulsive LJ forces.
template <class Awareness>
class PairwiseRepulsiveLJ : public PairwiseKernel
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using HandlerType  = PairwiseRepulsiveLJHandler<Awareness>; ///< Corresponding handler
    using ParamsType   = RepulsiveLJParams;                     ///< Corresponding parameters type
    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseRepulsiveLJ(real rc, real epsilon, real sigma, real maxForce, Awareness awareness) :
        rc_(rc),
        sigma_(sigma),
        maxForce_(maxForce),
        epsilon_(epsilon),
        awareness_(awareness),
        handler_(rc, epsilon, sigma, maxForce, awareness)
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

    virtual ~PairwiseRepulsiveLJ() = default;

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return handler_;
    }

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2,
               __UNUSED CellList *cl1, __UNUSED CellList *cl2, __UNUSED const MirState *state) override
    {
        awareness_.setup(lpv1, lpv2);
        handler_ = HandlerType(rc_, epsilon_, sigma_, maxForce_, awareness_);
    }

protected:
    real rc_;               ///< cutoff radius
    real sigma_;            ///< sigma parameter of the LJ potential
    real maxForce_;         ///< cutoff radius
    real epsilon_;          ///< energy coefficient
    Awareness awareness_;   ///< filtering of the force

    HandlerType handler_;   ///< GPU-compatible functor
};


/// Kernel for growing repulsive LJ forces.
template <class Awareness>
class PairwiseGrowingRepulsiveLJ : public PairwiseRepulsiveLJ<Awareness>
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using HandlerType  = PairwiseRepulsiveLJHandler<Awareness>; ///< Corresponding handler
    using ParamsType   = GrowingRepulsiveLJParams;              ///< Corresponding parameters type
    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseGrowingRepulsiveLJ(real rc, real epsilon, real sigma, real maxForce, Awareness awareness,
                               real initialLengthFraction, real growUntil) :
        PairwiseRepulsiveLJ<Awareness>(rc, epsilon, sigma, maxForce, std::move(awareness)),
        initLengthFraction_(initialLengthFraction),
        growUntil_(growUntil)
    {
        if (initialLengthFraction < 0 || initialLengthFraction > 1)
        {
            die("Wrong value of 'initialLengthFraction'. Must be in [0,1], got %g.", initialLengthFraction);
        }
    }

    /// Generic constructor
    PairwiseGrowingRepulsiveLJ(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        PairwiseGrowingRepulsiveLJ{rc,
                                   p.epsilon,
                                   p.sigma,
                                   p.maxForce,
                                   mpark::get<typename Awareness::ParamsType>(p.varAwarenessParams),
                                   p.initialLengthFraction,
                                   p.growUntil}
    {}


    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return this->handler_;
    }

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2,
               __UNUSED CellList *cl1, __UNUSED CellList *cl2, __UNUSED const MirState *state) override
    {
        const real l = _scaleFromTime(state->currentTime);
        this->awareness_.setup(lpv1, lpv2);
        this->handler_ = HandlerType(this->rc_,
                                     this->epsilon_ * l*l,
                                     this->sigma_ * l,
                                     this->maxForce_ * l,
                                     this->awareness_);
    }

private:

    /** Get a scaling factor for transforming the length scale of all the parameters
        (see also rescaleParameters())
        \param [in] t The simulation time (must be positive)
        \return scaling factor for length

        Will grow linearly from initLengthFraction_ to 1 during the first growUntil_ time interval.
        Otherwise this returns 1.
    */
    real _scaleFromTime(real t) const
    {
        return math::min(1.0_r, initLengthFraction_ + (1.0_r - initLengthFraction_) * (t / growUntil_));
    }


protected:
    real initLengthFraction_; ///< initial length scaling factor
    real growUntil_;          ///< time of initial growth
};

} // namespace mirheo
