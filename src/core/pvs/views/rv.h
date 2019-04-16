#pragma once

#include "ov.h"

#include "../rod_vector.h"

struct RVview : public OVview
{
    float4 *bishopQuaternions  { nullptr };
    float3 *bishopFrames       { nullptr };

    int nSegments { 0 };

    RVview(RodVector *rv = nullptr, LocalRodVector* lrv = nullptr, cudaStream_t stream = 0) :
        OVview(rv, lrv)
    {
        if (rv == nullptr || lrv == nullptr) return;

        bishopQuaternions  = lrv->bishopQuaternions.devPtr();
        bishopFrames       = lrv->bishopFrames.devPtr();
        nSegments          = lrv->getNumSegmentsPerRod();
    }
};

