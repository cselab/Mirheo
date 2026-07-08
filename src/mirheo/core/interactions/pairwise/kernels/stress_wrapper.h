// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/forceStress.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/mirheo_state.h>

#include <type_traits>

namespace mirheo {

class LocalParticleVector;
class CellList;

/** \brief Compute force and stress from a pairwise force kernel
    \tparam BasePairwiseForceHandler The underlying pairwise interaction handler (must output a force)
 */
template<typename BasePairwiseForceHandler>
class PairwiseStressWrapperHandler : public BasePairwiseForceHandler
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using BaseViewType = typename BasePairwiseForceHandler::ViewType;
    using ViewType      = PVviewWithStresses<BaseViewType>;
    using ParticleType  = typename BasePairwiseForceHandler::ParticleType;

    using BaseAccumulatorType = typename std::result_of<decltype(&BasePairwiseForceHandler::getZeroedAccumulator)(BasePairwiseForceHandler)>::type;
    using BaseForceType = decltype(BaseAccumulatorType{}.get());
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseStressWrapperHandler(BasePairwiseForceHandler basicForceHandler) :
        BasePairwiseForceHandler(basicForceHandler)
    {}

    /// Evaluate the force and the stress
    __device__ inline auto operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = getPosition(dst) - getPosition(src);
        const auto force  = BasePairwiseForceHandler::operator()(dst, dstId, src, srcId);
        const real3 f = getForce(force);
        Stress s;

        s.xx = 0.5_r * dr.x * f.x;
        s.xy = 0.5_r * dr.x * f.y;
        s.xz = 0.5_r * dr.x * f.z;
        s.yy = 0.5_r * dr.y * f.y;
        s.yz = 0.5_r * dr.y * f.z;
        s.zz = 0.5_r * dr.z * f.z;

        return ForceStress<BaseForceType>{force, s};
    }

    /// Initialize the accumulator
    __D__ inline auto getZeroedAccumulator() const {return ForceStressAccumulator<BaseViewType,BaseAccumulatorType>();}
};

/** \brief Create PairwiseStressWrapperHandler from host
    \tparam BasePairwiseForceHandler The underlying pairwise interaction (must output a force)
 */
template<typename BasePairwiseForce>
class PairwiseStressWrapper : public BasePairwiseForce
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using BaseHandlerType = typename BasePairwiseForce::HandlerType;
    using HandlerType  = PairwiseStressWrapperHandler< BaseHandlerType >;

    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;
    using ParamsType   = typename BasePairwiseForce::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseStressWrapper(BasePairwiseForce basicForce) :
        BasePairwiseForce(basicForce),
        basicForceWrapperHandler_(basicForce.handler())
    {}

    /// Generic constructor
    PairwiseStressWrapper(real rc, const ParamsType& p, long seed=42424242) :
        BasePairwiseForce(rc, p, seed),
        basicForceWrapperHandler_ {HandlerType{BasePairwiseForce::handler()}}
    {}

    /// setup internal state
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const MirState *state) override
    {
        BasePairwiseForce::setup(lpv1, lpv2, cl1, cl2, state);
        basicForceWrapperHandler_ = HandlerType(BasePairwiseForce::handler());
    }

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return basicForceWrapperHandler_;
    }

private:
    HandlerType basicForceWrapperHandler_;
};

} // namespace mirheo
