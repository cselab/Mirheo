#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/forceStress.h>
#include <core/utils/common.h>
#include <core/ymero_state.h>

#include <type_traits>

class LocalParticleVector;
class CellList;

template<typename BasicPairwiseForceHandler>
class PairwiseStressWrapperHandler : public BasicPairwiseForceHandler
{
public:

    using BasicViewType = typename BasicPairwiseForceHandler::ViewType;
    using ViewType      = PVviewWithStresses<BasicViewType>;
    using ParticleType  = typename BasicPairwiseForceHandler::ParticleType;
    
    PairwiseStressWrapperHandler(BasicPairwiseForceHandler basicForceHandler) :
        BasicPairwiseForceHandler(basicForceHandler)
    {}
    
    __device__ inline ForceStress operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {        
        float3 dr = getPosition(dst) - getPosition(src);
        float3 f  = BasicPairwiseForceHandler::operator()(dst, dstId, src, srcId);
        Stress s;
        
        s.xx = 0.5f * dr.x * f.x;
        s.xy = 0.5f * dr.x * f.y;
        s.xz = 0.5f * dr.x * f.z;
        s.yy = 0.5f * dr.y * f.y;
        s.yz = 0.5f * dr.y * f.z;
        s.zz = 0.5f * dr.z * f.z;        

        return {f, s};
    }

    __D__ inline ForceStressAccumulator<BasicViewType> getZeroedAccumulator() const {return ForceStressAccumulator<BasicViewType>();}
};

template<typename BasicPairwiseForce>
class PairwiseStressWrapper : public BasicPairwiseForce
{
public:

    using BasicHandlerType = typename BasicPairwiseForce::HandlerType;
    using HandlerType  = PairwiseStressWrapperHandler< BasicHandlerType >;

    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;

    PairwiseStressWrapper(BasicPairwiseForce basicForce) :
        BasicPairwiseForce(basicForce),
        basicForceWrapperHandler(basicForce.handler())
    {}

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state) override
    {
        BasicPairwiseForce::setup(lpv1, lpv2, cl1, cl2, state);
        basicForceWrapperHandler = HandlerType(BasicPairwiseForce::handler());
    }

    const HandlerType& handler() const
    {
        return basicForceWrapperHandler;
    }
    
protected:
    HandlerType basicForceWrapperHandler;
};
