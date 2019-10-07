#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"

#include <core/mirheo_state.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rod_vector.h>

struct LJAwarenessNone
{
    LJAwarenessNone() = default;
    void setup(__UNUSED LocalParticleVector *lpv1, __UNUSED LocalParticleVector *lpv2) {}
    __D__ inline bool interact(__UNUSED int srcId, __UNUSED int dstId) const {return true;}
};

struct LJAwarenessObject
{
    LJAwarenessObject() = default;
    
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

                if (abs(srcSegId - dstSegId) <= minSegmentsDist)
                    return false;
            }
        }
        return true;
    }

    bool self {false};
    int objSize {0}, minSegmentsDist{0};
};

template <class Awareness>
class PairwiseLJ : public PairwiseKernel, public ParticleFetcher
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseLJ;
    
    PairwiseLJ(float rc, float epsilon, float sigma, float maxForce, Awareness awareness) :
        ParticleFetcher(rc),
        epsilon(epsilon),
        sigma(sigma),
        maxForce(maxForce),
        epsx24_sigma(24.0 * epsilon / sigma),
        awareness(awareness)
    {}

    __D__ inline float3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        if (!awareness.interact(src.i1, dst.i1))
            return make_float3(0.0f);
        
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);

        if (rij2 > rc2) return make_float3(0.0f);

        const float rs2 = sigma*sigma / rij2;
        const float rs4 = rs2*rs2;
        const float rs8 = rs4*rs4;
        const float rs14 = rs8*rs4*rs2;

        const float IfI = epsx24_sigma * (2*rs14 - rs8);

        return dr * min(max(IfI, 0.0f), maxForce);
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

    float epsilon, sigma, maxForce;
    float epsx24_sigma;

    Awareness awareness;
};
