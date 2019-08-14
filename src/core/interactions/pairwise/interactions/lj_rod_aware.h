#pragma once

#include "lj.h"

#include <core/datatypes.h>
#include <core/pvs/rod_vector.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>
#include <core/mirheo_state.h>

class PairwiseLJRodAware : public PairwiseLJ
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseLJRodAware;
    
    PairwiseLJRodAware(float rc, float epsilon, float sigma, float maxForce, int minSegmentsDist) :
        PairwiseLJ(rc, epsilon, sigma, maxForce),
        minSegmentsDist(minSegmentsDist)
    {}

    __D__ inline float3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        if (self)
        {
            int dstObjId = dst.i1 / objSize;
            int srcObjId = src.i1 / objSize;

            if (dstObjId == srcObjId)
            {
                int srcSegId = (dst.i1 % objSize) / 5;
                int dstSegId = (src.i1 % objSize) / 5;

                if (abs(srcSegId - dstSegId) <= minSegmentsDist)
                    return make_float3(0.0f);
            }
        }

        return PairwiseLJ::operator() (dst, dstId, src, srcId);
    }

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const MirState *state) override
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

protected:

    bool self {false};

    int objSize {0}, minSegmentsDist;
};

