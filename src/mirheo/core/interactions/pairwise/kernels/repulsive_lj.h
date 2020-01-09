#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

struct LJAwarenessNone
{
    LJAwarenessNone() = default;
    LJAwarenessNone(__UNUSED const LJAwarenessParamsNone& params) {}
    
    void setup(__UNUSED LocalParticleVector *lpv1, __UNUSED LocalParticleVector *lpv2) {}
    __D__ inline bool interact(__UNUSED int srcId, __UNUSED int dstId) const {return true;}
};

struct LJAwarenessObject
{
    LJAwarenessObject() = default;
    LJAwarenessObject(__UNUSED const LJAwarenessParamsObject& params) {}
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2)
    {
        auto ov1 = dynamic_cast<ObjectVector*>(lpv1->pv);
        auto ov2 = dynamic_cast<ObjectVector*>(lpv2->pv);

        self = false;
        if (ov1 != nullptr && ov2 != nullptr && lpv1 == lpv2)
        {
            self = true;
            objSize = ov1->objSize;
        }
    }

    __D__ inline bool interact(int srcId, int dstId) const
    {
        if (self)
        {
            const int dstObjId = dstId / objSize;
            const int srcObjId = srcId / objSize;

            if (dstObjId == srcObjId)
                return false;
        }
        return true;
    }
        
    bool self {false};
    int objSize {0};
};

struct LJAwarenessRod
{
    LJAwarenessRod(int minSegmentsDist) :
        minSegmentsDist(minSegmentsDist)
    {}

    LJAwarenessRod(const LJAwarenessParamsRod& params) :
        LJAwarenessRod(params.minSegmentsDist)
    {}

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2)
    {
        auto rv1 = dynamic_cast<RodVector*>(lpv1->pv);
        auto rv2 = dynamic_cast<RodVector*>(lpv2->pv);

        self = false;
        if (rv1 != nullptr && rv2 != nullptr && lpv1 == lpv2)
        {
            self = true;
            objSize = rv1->objSize;
        }
    }

    __D__ inline bool interact(int srcId, int dstId) const
    {
        if (self)
        {
            const int dstObjId = dstId / objSize;
            const int srcObjId = srcId / objSize;

            if (dstObjId == srcObjId)
            {
                const int srcSegId = (dstId % objSize) / 5;
                const int dstSegId = (srcId % objSize) / 5;

                if (math::abs(srcSegId - dstSegId) <= minSegmentsDist)
                    return false;
            }
        }
        return true;
    }

    bool self {false};
    int objSize {0}, minSegmentsDist{0};
};

template <class Awareness>
class PairwiseRepulsiveLJ : public PairwiseKernel, public ParticleFetcher
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseRepulsiveLJ;
    
    PairwiseRepulsiveLJ(real rc, real epsilon, real sigma, real maxForce, Awareness awareness) :
        ParticleFetcher(rc),
        epsilon(epsilon),
        sigma(sigma),
        maxForce(maxForce),
        epsx24_sigma2(24.0_r * epsilon / (sigma * sigma)),
        awareness(awareness)
    {}

    __D__ inline real3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        constexpr real tolerance = 1e-6_r;
        if (!awareness.interact(src.i1, dst.i1))
            return make_real3(0.0_r);
        
        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);

        if (dr2 > rc2 || dr2 < tolerance)
            return make_real3(0.0_r);

        const real rs2 = sigma*sigma / dr2;
        const real rs4 = rs2*rs2;
        const real rs8 = rs4*rs4;
        const real rs14 = rs8*rs4*rs2;

        const real IfI = epsx24_sigma2 * (2*rs14 - rs8);

        return dr * math::min(math::max(IfI, 0.0_r), maxForce);
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2,
               __UNUSED CellList *cl1, __UNUSED CellList *cl2, __UNUSED const MirState *state) override
    {
        awareness.setup(lpv1, lpv2);
    }

    
private:

    real epsilon, sigma, maxForce;
    real epsx24_sigma2;

    Awareness awareness;
};

} // namespace mirheo
