#include "interface.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

Interaction::Interaction(const MirState *state, std::string name) :
    MirSimulationObject(state, name)
{}
Interaction::Interaction(const MirState *state, Loader& loader, const ConfigObject& config) :
    MirSimulationObject(state, loader, config)
{
    // Note: we do NOT load the `impl` object here. Since impl typically has
    // template arguments, loading must be done from the derived class. For
    // example, see the MembraneInteraction's constructor and variantForeach.
}

Interaction::~Interaction() = default;

void Interaction::setPrerequisites(__UNUSED ParticleVector *pv1,
                                   __UNUSED ParticleVector *pv2,
                                   __UNUSED CellList *cl1,
                                   __UNUSED CellList *cl2)
{}

std::vector<Interaction::InteractionChannel> Interaction::getInputChannels() const
{
    return {};
}

std::vector<Interaction::InteractionChannel> Interaction::getOutputChannels() const
{
    return {{ChannelNames::forces, alwaysActive}};
}

void Interaction::setState(const MirState *state)
{
    MirSimulationObject::setState(state);
    if (impl) impl->setState(state);
}

bool Interaction::isSelfObjectInteraction() const
{
    return false;
}

void Interaction::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    if (!impl) return;
    impl->checkpoint(comm, path, checkpointId);
}

void Interaction::restart(MPI_Comm comm, const std::string& path)
{
    if (!impl) return;
    impl->restart(comm, path);
}

real Interaction::getCutoffRadius() const
{
    return 1.0_r;
}

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};

ConfigObject Interaction::_saveSnapshotWithoutImpl(Saver& saver, const std::string& typeName)
{
    return MirSimulationObject::_saveSnapshot(saver, "Interaction", typeName);
}

ConfigObject Interaction::_saveSnapshotWithImpl(Saver& saver, const std::string& typeName)
{
    ConfigObject config = _saveSnapshotWithoutImpl(saver, typeName);
    config.emplace("impl", saver(impl));
    return config;
}

ConfigObject Interaction::_saveImplSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = MirSimulationObject::_saveSnapshot(saver, "InteractionImpl", typeName);
    if (impl)
        die("Impl interaction has impl?");
    return config;
}

} // namespace mirheo
