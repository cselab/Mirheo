// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "awareness.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

namespace mirheo
{

/** \brief Compute Morse forces on the device.
    \tparam Awareness A functor that describes which particles pairs interact.
 */
template <class Awareness>
class PairwiseMorse : public PairwiseKernel, public ParticleFetcher
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using ViewType     = PVview;        ///< Compatible view type
    using ParticleType = Particle;      ///< Compatible particle type
    using HandlerType  = PairwiseMorse<Awareness>; ///< Corresponding handler
    using ParamsType   = MorseParams;   ///< Corresponding parameters type
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseMorse(real rc, real De, real r0, real beta, Awareness awareness) :
        ParticleFetcher(rc),
        twoDeBeta_(2 * De * beta),
        r0_(r0),
        beta_(beta),
        awareness_(awareness)
    {}

    /// Generic constructor
    PairwiseMorse(real rc, const ParamsType& p, __UNUSED long seed) :
        PairwiseMorse{rc,
                      p.De, p.r0, p.beta,
                      std::get<typename Awareness::ParamsType>(p.varAwarenessParams)}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, __UNUSED int dstId,
                                  ParticleType src, __UNUSED int srcId) const
    {
        if (!awareness_.interact(src.i1, dst.i1))
            return make_real3(0.0_r);

        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);
        if (dr2 > rc2_)
            return make_real3(0.0_r);

        const real r = math::sqrt(dr2);
        const real expTerm = math::exp(beta_ * (r0_ - r));
        const real magn = twoDeBeta_ * expTerm * (expTerm - 1.0_r);

        const real3 er = dr / math::max(r, 1e-6_r);
        return magn * er;
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
        return "PairwiseMorse";
    }

private:
    real twoDeBeta_; ///< 2 * De * beta
    real r0_;
    real beta_;
    Awareness awareness_;
};

} // namespace mirheo
