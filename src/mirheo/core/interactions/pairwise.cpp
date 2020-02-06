#include "pairwise.h"

#include <mirheo/core/utils/config.h>

namespace mirheo
{

ConfigDictionary PairwiseInteraction::writeSnapshot(Dumper& dumper)
{
    return {
        {"__category",      dumper("Interaction")},
        {"__type",          dumper("PairwiseInteraction")},
        {"varParams",       dumper(varParams)},
        {"varStressParams", dumper(varStressParams)},
    };
}

} // namespace mirheo
