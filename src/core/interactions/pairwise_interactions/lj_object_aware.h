#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/utils/cuda_common.h>

class Pairwise_LJObjectAware
{
public:
    Pairwise_LJObjectAware(float rc, float sigma, float epsilon, float maxForce) :
        lj(rc, sigma, epsilon, maxForce)
    {    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
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

    __device__ inline float3 operator()(Particle dst, int dstId, Particle src, int srcId) const
    {
        if (self)
        {
            const int dstObjId = dst.i1 / objSize;
            const int srcObjId = src.i1 / objSize;

            if (dstObjId == srcObjId) return make_float3(0.0f);
        }

        float3 f = lj(dst, dstId, src, srcId);

        return f;
    }


private:

    bool self;

    int objSize;

    Pairwise_LJ lj;
};
