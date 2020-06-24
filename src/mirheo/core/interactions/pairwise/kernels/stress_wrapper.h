// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/forceStress.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/mirheo_state.h>

#include <type_traits>

namespace mirheo
{

class LocalParticleVector;
class CellList;

/** \brief Compute force and stress from a pairwise force kernel
    \tparam BasicPairwiseForceHandler The underlying pairwise interaction handler (must output a force)
 */
template<typename BasicPairwiseForceHandler>
class PairwiseStressWrapperHandler : public BasicPairwiseForceHandler
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using BasicViewType = typename BasicPairwiseForceHandler::ViewType;
    using ViewType      = PVviewWithStresses<BasicViewType>;
    using ParticleType  = typename BasicPairwiseForceHandler::ParticleType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseStressWrapperHandler(BasicPairwiseForceHandler basicForceHandler) :
        BasicPairwiseForceHandler(basicForceHandler)
    {}

    /// Evaluate the force and the stress
    __device__ inline ForceStress operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = getPosition(dst) - getPosition(src);
        const real3 f  = BasicPairwiseForceHandler::operator()(dst, dstId, src, srcId);
        Stress s;

        s.xx = 0.5_r * dr.x * f.x;
        s.xy = 0.5_r * dr.x * f.y;
        s.xz = 0.5_r * dr.x * f.z;
        s.yy = 0.5_r * dr.y * f.y;
        s.yz = 0.5_r * dr.y * f.z;
        s.zz = 0.5_r * dr.z * f.z;

        return {f, s};
    }

    /// Initialize the accumulator
    __D__ inline ForceStressAccumulator<BasicViewType> getZeroedAccumulator() const {return ForceStressAccumulator<BasicViewType>();}
};

/** \brief Create PairwiseStressWrapperHandler from host
    \tparam BasicPairwiseForceHandler The underlying pairwise interaction (must output a force)
 */
template<typename BasicPairwiseForce>
class PairwiseStressWrapper : public BasicPairwiseForce
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using BasicHandlerType = typename BasicPairwiseForce::HandlerType;
    using HandlerType  = PairwiseStressWrapperHandler< BasicHandlerType >;

    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;
    using ParamsType   = typename BasicPairwiseForce::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseStressWrapper(BasicPairwiseForce basicForce) :
        BasicPairwiseForce(basicForce),
        basicForceWrapperHandler_(basicForce.handler())
    {}

    /// Generic constructor
    PairwiseStressWrapper(real rc, const ParamsType& p, real dt, long seed=42424242) :
        BasicPairwiseForce(rc, p, dt, seed),
        basicForceWrapperHandler_ {HandlerType{BasicPairwiseForce::handler()}}
    {}

    /// setup internal state
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const MirState *state) override
    {
        BasicPairwiseForce::setup(lpv1, lpv2, cl1, cl2, state);
        basicForceWrapperHandler_ = HandlerType(BasicPairwiseForce::handler());
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
