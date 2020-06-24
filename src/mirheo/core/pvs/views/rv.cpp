// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "rv.h"

#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

RVview::RVview(RodVector *rv, LocalRodVector *lrv) :
    OVview(rv, lrv)
{
    nSegments = lrv->getNumSegmentsPerRod();

    auto& data = lrv->dataPerBisegment;

    if (data.checkChannelExists(channel_names::polyStates))
        states = data.getData<int>(channel_names::polyStates)->devPtr();

    if (data.checkChannelExists(channel_names::energies))
        energies = data.getData<real>(channel_names::energies)->devPtr();
}

RVviewWithOldParticles::RVviewWithOldParticles(RodVector *rv, LocalRodVector *lrv) :
    RVview(rv, lrv)
{
    oldPositions = lrv->dataPerParticle.getData<real4>(channel_names::oldPositions)->devPtr();
}

} // namespace mirheo
