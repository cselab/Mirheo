#include "rv.h"

#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

RVview::RVview(RodVector *rv, LocalRodVector *lrv) :
    OVview(rv, lrv)
{
    nSegments = lrv->getNumSegmentsPerRod();

    auto& data = lrv->dataPerBisegment;
        
    if (data.checkChannelExists(ChannelNames::polyStates))
        states = data.getData<int>(ChannelNames::polyStates)->devPtr();
        
    if (data.checkChannelExists(ChannelNames::energies))
        energies = data.getData<real>(ChannelNames::energies)->devPtr();
}

RVviewWithOldParticles::RVviewWithOldParticles(RodVector *rv, LocalRodVector *lrv) :
    RVview(rv, lrv)
{
    oldPositions = lrv->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->devPtr();
}

} // namespace mirheo
