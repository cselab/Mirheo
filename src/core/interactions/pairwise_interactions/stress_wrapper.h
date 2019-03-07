#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/forceStress.h>
#include <core/utils/common.h>
#include <core/ymero_state.h>

#include <type_traits>

class LocalParticleVector;
class CellList;

template<typename BasicPairwiseForceHandler>
class PairwiseStressWrapperHandler
{
public:

    using BasicViewType = typename BasicPairwiseForceHandler::ViewType;
    using ViewType      = PVviewWithStresses<BasicViewType>;
    using ParticleType  = typename BasicPairwiseForceHandler::ParticleType;
    
    PairwiseStressWrapperHandler(BasicPairwiseForceHandler basicForceHandler) :
        basicForceHandler(basicForceHandler)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const                     { return        basicForceHandler.read(view, id); }
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const              { return basicForceHandler.readNoCache(view, id); }
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const { basicForceHandler.readCoordinates(p, view, id); }
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { basicForceHandler.readExtraData  (p, view, id); }
    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const { return basicForceHandler.withinCutoff(src, dst);}
    __D__ inline float3 getPosition(const ParticleType& p) const {return basicForceHandler.getPosition(p);}
    
    __device__ inline ForceStress operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {        
        float3 dr = getPosition(dst) - getPosition(src);
        float3 f  = basicForceHandler(dst, dstId, src, srcId);
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

private:
    
    BasicPairwiseForceHandler basicForceHandler;
};

template<typename BasicPairwiseForce>
class PairwiseStressWrapper
{
public:

    using BasicHandlerType = typename BasicPairwiseForce::HandlerType;
    using HandlerType  = PairwiseStressWrapperHandler< BasicHandlerType >;

    using ViewType     = typename HandlerType::ViewType;
    using ParticleType = typename HandlerType::ParticleType;

    PairwiseStressWrapper(BasicPairwiseForce basicForce) :
        basicForce(basicForce),
        basicForceWrapperHandler(basicForce.handler())
    {}

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state)
    {
        basicForce.setup(lpv1, lpv2, cl1, cl2, state);
        basicForceWrapperHandler = HandlerType(basicForce.handler());
    }

    const HandlerType& handler() const
    {
        return basicForceWrapperHandler;
    }
    
protected:
    BasicPairwiseForce basicForce;
    HandlerType basicForceWrapperHandler;
};
