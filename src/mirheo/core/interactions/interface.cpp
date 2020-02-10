#include "interface.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

Interaction::Interaction(const MirState *state, std::string name, real rc) :
    MirSimulationObject(state, name),
    rc(rc)
{}
Interaction::Interaction(const MirState *state, Undumper& undumper, const ConfigDictionary& config) :
    MirSimulationObject(state, undumper, config),
    rc(config["rc"])
{
    // Note: we do NOT load the `impl` object here. Since impl typically has
    // template arguments, that must be done from the derived class. See the
    // implementation of MembraneInteraction's constructor using
    // variantForeach.
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

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};

ConfigDictionary Interaction::_saveSnapshotWithoutImpl(Dumper& dumper, const std::string& typeName)
{
    ConfigDictionary dict = MirSimulationObject::_saveSnapshot(dumper, "Interaction", typeName);
    dict.emplace("rc", dumper(rc));
    return dict;
}

ConfigDictionary Interaction::_saveSnapshotWithImpl(Dumper& dumper, const std::string& typeName)
{
    ConfigDictionary dict = _saveSnapshotWithoutImpl(dumper, typeName);
    dict.emplace("impl", dumper(impl));
    return dict;
}

ConfigDictionary Interaction::_saveImplSnapshot(Dumper& dumper, const std::string& typeName)
{
    ConfigDictionary dict = MirSimulationObject::_saveSnapshot(dumper, "InteractionImpl", typeName);
    dict.emplace("rc", dumper(rc));
    if (impl)
        die("Impl interaction has impl?");
    return dict;
}


} // namespace mirheo
