#pragma once

#include "lj.h"

#include <core/datatypes.h>
#include <core/interactions/accumulators/force.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>
#include <core/ymero_state.h>

class PairwiseLJObjectAware : public PairwiseLJ
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    using HandlerType  = PairwiseLJObjectAware;
    
    PairwiseLJObjectAware(float rc, float epsilon, float sigma, float maxForce) :
        PairwiseLJ(rc, epsilon, sigma, maxForce)
    {}

    __D__ inline float3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        if (self)
        {
            const int dstObjId = dst.i1 / objSize;
            const int srcObjId = src.i1 / objSize;

            if (dstObjId == srcObjId) return make_float3(0.0f);
        }

        return PairwiseLJ::operator() (dst, dstId, src, srcId);
    }

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, const YmrState *state)
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

protected:

    bool self;

    int objSize;
};

