#include "base_pairwise.h"

#include <mirheo/core/utils/config.h>

namespace mirheo
{

BasePairwiseInteraction::BasePairwiseInteraction(const MirState *state, const std::string& name, real rc) :
    Interaction(state, name),
    rc_(rc)
{}

BasePairwiseInteraction::BasePairwiseInteraction(const MirState *state, __UNUSED Loader& loader, const ConfigObject& config) :
    BasePairwiseInteraction{state,
                            config["name"],
                            config["rc"]}
{}

BasePairwiseInteraction::~BasePairwiseInteraction() = default;
    
real BasePairwiseInteraction::getCutoffRadius() const
{
    return rc_;
}

void BasePairwiseInteraction::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<BasePairwiseInteraction>(
            this, _saveSnapshot(saver, "BasePairwiseInteraction"));
}

ConfigObject BasePairwiseInteraction::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Interaction::_saveSnapshotWithoutImpl(saver, typeName);
    config.emplace("rc", saver(getCutoffRadius()));
    return config;
}

} // namespace mirheo
