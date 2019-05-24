#pragma once

#include "ov.h"

#include "../rod_vector.h"

struct RVview : public OVview
{
    int nSegments { 0 };

    RVview(RodVector *rv = nullptr, LocalRodVector *lrv = nullptr, cudaStream_t stream = 0) :
        OVview(rv, lrv)
    {
        if (rv == nullptr || lrv == nullptr) return;
        nSegments          = lrv->getNumSegmentsPerRod();
    }
};

