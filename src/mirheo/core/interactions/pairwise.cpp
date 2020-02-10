#include "pairwise.h"

#include <mirheo/core/utils/config.h>

namespace mirheo
{

PairwiseInteraction::PairwiseInteraction(const MirState *state, Undumper& un,
                                         const ConfigDictionary& dict) :
    PairwiseInteraction{
        state, dict["name"], dict["rc"],
        un.undump<VarPairwiseParams>(dict["varParams"]),
        un.undump<VarStressParams>(dict["varStressParams"]),
    }
{
    assert(dict["__type"].getString() == "PairwiseInteraction");
}

void PairwiseInteraction::saveSnapshotAndRegister(Dumper& dumper)
{
    dumper.registerObject<PairwiseInteraction>(
            this, _saveSnapshot(dumper, "PairwiseInteraction"));
}

ConfigDictionary PairwiseInteraction::_saveSnapshot(Dumper& dumper, const std::string& typeName)
{
    ConfigDictionary dict = Interaction::_saveSnapshotWithoutImpl(dumper, typeName);
    dict.emplace("varParams",       dumper(varParams));
    dict.emplace("varStressParams", dumper(varStressParams));
    return dict;
}

} // namespace mirheo
