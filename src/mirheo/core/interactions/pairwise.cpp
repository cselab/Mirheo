#include "pairwise.h"

#include <mirheo/core/utils/config.h>

namespace mirheo
{

PairwiseInteraction::PairwiseInteraction(const MirState *state, Undumper& un,
                                         const ConfigDictionary& dict) :
    PairwiseInteraction{
        state,
        un.undump<std::string>(dict.at("name")),
        un.undump<real>(dict.at("rc")),
        un.undump<VarPairwiseParams>(dict.at("varParams")),
        un.undump<VarStressParams>(dict.at("varStressParams")),
    }
{
    assert(dict.at("__type").getString() == "PairwiseInteraction");
}

ConfigDictionary PairwiseInteraction::writeSnapshot(Dumper& dumper)
{
    return {
        {"__category",      dumper("Interaction")},
        {"__type",          dumper("PairwiseInteraction")},
        {"rc",              dumper(rc)},
        {"varParams",       dumper(varParams)},
        {"varStressParams", dumper(varStressParams)},
    };
}

} // namespace mirheo
