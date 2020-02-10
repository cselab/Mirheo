#include "pairwise.h"

#include <mirheo/core/utils/config.h>

namespace mirheo
{

PairwiseInteraction::PairwiseInteraction(const MirState *state, Undumper& un,
                                         const ConfigObject& config) :
    PairwiseInteraction{
        state, config["name"], config["rc"],
        un.undump<VarPairwiseParams>(config["varParams"]),
        un.undump<VarStressParams>(config["varStressParams"]),
    }
{
    assert(config["__type"].getString() == "PairwiseInteraction");
}

void PairwiseInteraction::saveSnapshotAndRegister(Dumper& dumper)
{
    dumper.registerObject<PairwiseInteraction>(
            this, _saveSnapshot(dumper, "PairwiseInteraction"));
}

ConfigObject PairwiseInteraction::_saveSnapshot(Dumper& dumper, const std::string& typeName)
{
    ConfigObject config = Interaction::_saveSnapshotWithoutImpl(dumper, typeName);
    config.emplace("varParams",       dumper(varParams));
    config.emplace("varStressParams", dumper(varStressParams));
    return config;
}

} // namespace mirheo
