#pragma once

#include "ov.h"

#include "../rod_vector.h"

struct RVview : public OVview
{
    int   nSegments { 0 };
    int   *states   { nullptr };
    float *energies { nullptr };

    RVview(RodVector *rv = nullptr, LocalRodVector *lrv = nullptr, cudaStream_t stream = 0) :
        OVview(rv, lrv)
    {
        if (rv == nullptr || lrv == nullptr) return;
        nSegments          = lrv->getNumSegmentsPerRod();

        auto& data = lrv->dataPerParticle;
        
        if (data.checkChannelExists(ChannelNames::polyStates))
            states = data.getData<int>(ChannelNames::polyStates)->devPtr();

        if (data.checkChannelExists(ChannelNames::energies))
            energies = data.getData<float>(ChannelNames::energies)->devPtr();
    }
};

