#pragma once

#include "ov.h"

#include "../rod_vector.h"

struct RVview : public OVview
{
    int   nSegments {0};
    int   *states   {nullptr};
    float *energies {nullptr};

    RVview(RodVector *rv, LocalRodVector *lrv) :
        OVview(rv, lrv)
    {
        nSegments = lrv->getNumSegmentsPerRod();

        auto& data = lrv->dataPerBisegment;
        
        if (data.checkChannelExists(ChannelNames::polyStates))
            states = data.getData<int>(ChannelNames::polyStates)->devPtr();
        
        if (data.checkChannelExists(ChannelNames::energies))
            energies = data.getData<float>(ChannelNames::energies)->devPtr();
    }
};

struct RVviewWithOldParticles : public RVview
{
    float4 *oldPositions {nullptr};

    RVviewWithOldParticles(RodVector *rv, LocalRodVector *lrv) :
        RVview(rv, lrv)
    {
        oldPositions = lrv->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->devPtr();
    }
};
