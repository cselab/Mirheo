// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"
#include "factory_helper.h"

#include "density.h"
#include "dpd.h"
#include "dpd_visco_elastic.h"
#include "growing_repulsive_lj.h"
#include "lj.h"
#include "mdpd.h"
#include "morse.h"
#include "repulsive_lj.h"
#include "sdpd.h"

namespace mirheo {

std::unique_ptr<BasePairwiseInteraction>
createInteractionPairwise(const MirState *state, const std::string& name, real rc,
                          const std::string& type, ParametersWrap& desc)
{
    std::unique_ptr<BasePairwiseInteraction> interaction;


    if (type == "DPD")
    {
        const auto params = factory_helper::readDPDParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = std::make_unique<PairwiseDPDInteraction>(state, name, rc, params, stressPeriod);
    }
    else if (type == "MDPD")
    {
        const auto params = factory_helper::readMDPDParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = std::make_unique<PairwiseMDPDInteraction>(state, name, rc, params, stressPeriod);
    }
    else if (type == "SDPD")
    {
        const auto params = factory_helper::readSDPDParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = makePairwiseSDPDInteraction(state, name, rc, params, stressPeriod);
    }
    else if (type == "ViscoElasticDPD")
    {
        const auto params = factory_helper::readViscoElasticDPDParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = std::make_unique<PairwiseViscoElasticDPDInteraction>(state, name, rc, params, stressPeriod);
    }
    else if (type == "LJ")
    {
        const auto params = factory_helper::readLJParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = std::make_unique<PairwiseLJInteraction>(state, name, rc, params, stressPeriod);
    }
    else if (type == "Morse")
    {
        const auto params = factory_helper::readMorseParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = makePairwiseMorseInteraction(state, name, rc, params, stressPeriod);
    }
    else if (type == "RepulsiveLJ")
    {
        const auto params = factory_helper::readRepulsiveLJParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = makePairwiseRepulsiveLJInteraction(state, name, rc, params, stressPeriod);
    }
    else if (type == "GrowingRepulsiveLJ")
    {
        const auto params = factory_helper::readGrowingRepulsiveLJParams(desc);
        const auto stressPeriod = factory_helper::readStressPeriod(desc);
        interaction = makePairwiseGrowingRepulsiveLJInteraction(state, name, rc, params, stressPeriod);
    }
    else if (type == "Density")
    {
        const auto params = factory_helper::readDensityParams(desc);
        interaction = makePairwiseDensityInteraction(state, name, rc, params);
    }
    else
    {
        die("Unrecognized pairwise interaction type '%s'", type.c_str());
    }

    desc.checkAllRead();

    return interaction;
}

} // namespace mirheo
