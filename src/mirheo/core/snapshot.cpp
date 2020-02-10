#include "snapshot.h"
#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/mesh/factory.h>
#include <mirheo/core/pvs/factory.h>
#include <mirheo/core/interactions/factory.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/folders.h>

#include <typeinfo>

namespace mirheo
{

void _unknownRefStringError [[noreturn]] (const std::string &name)
{
    die("Unknown object reference \"%s\".", name.c_str());
}
void _dynamicCastError [[noreturn]] (const char *from, const char *to)
{
    die("Dynamic cast error from runtime type \"%s\" to static type \"%s\".", from, to);
}


LoaderContext::LoaderContext(std::string path) :
    LoaderContext{configFromJSONFile(joinPaths(path, "config.compute.json")),
                  configFromJSONFile(joinPaths(path, "config.post.json")),
                  std::move(path)}
{}

LoaderContext::LoaderContext(ConfigValue compute, ConfigValue postprocess,
                             std::string snapshotPath) :
    path_{std::move(snapshotPath)},
    compConfig_{std::move(compute)},
    postConfig_{std::move(postprocess)}
{}

LoaderContext::~LoaderContext() = default;

const ConfigObject& LoaderContext::getCompObjectConfig(
        const std::string& category, const std::string& name)
{
    for (const ConfigValue& obj : getComp()[category].getArray())
        if (obj.getObject()["name"].getString() == name)
            return obj.getObject();
    die("Object category=\"%s\" name=\"%s\" not found.", category.c_str(), name.c_str());
}


void loadSnapshot(Mirheo *mir, Loader& loader)
{
    auto& context = loader.getContext();
    const ConfigValue& compConfig = context.getComp();

    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(context.getPath());

    auto loadObject = [mir, &loader](
            const ConfigValue& info,
            auto &objects,
            auto func) -> decltype(auto)
    {
        auto ptr = func(mir->getState(), loader, info.getObject(), info["__type"]);
        if (!ptr)
            die("Factory returned a nullptr for object: %s", info.toJSONString().c_str());
        auto it = objects.emplace(info["name"], std::move(ptr)).first;
        debug("Loaded the object \"%s\".\n", it->first.c_str());
        return it->second;
    };

    if (auto *meshInfos = compConfig.get("Mesh")) {
        auto &meshes = context.getContainerShared<Mesh>();
        for (const auto& info : meshInfos->getArray())
            loadObject(info, meshes, loadMesh);
    }

    if (auto *pvInfos = compConfig.get("ParticleVector")) {
        auto &pvs = context.getContainerShared<ParticleVector>();
        for (const auto& info : pvInfos->getArray()) {
            const auto &pv = loadObject(info, pvs, ParticleVectorFactory::loadParticleVector);
            mir->registerParticleVector(pv, ic);
        }
    }

    if (mir->isComputeTask()) {
        if (auto *interactionInfos = compConfig.get("Interaction")) {
            auto &interactions = context.getContainerShared<Interaction>();
            for (const auto& info : interactionInfos->getArray()) {
                const auto& interaction = loadObject(info, interactions, InteractionFactory::loadInteraction);
                mir->registerInteraction(interaction);
            }
        }

        const ConfigValue& sim = compConfig["Simulation"][0];
        if (auto *interactionPrototypes = sim.get("interactionPrototypes")) {
            for (const auto& info : interactionPrototypes->getArray()) {
                const auto& pv1 = context.getShared<ParticleVector>(info["pv1"]);
                const auto& pv2 = context.getShared<ParticleVector>(info["pv2"]);
                const auto& interaction = context.getShared<Interaction>(info["interaction"]);
                real rc = loader.load<real>(info["rc"]);
                if (rc != interaction->rc)
                    die("Interaction rc do not match: %f != %f.", rc, interaction->rc);
                mir->setInteraction(interaction.get(), pv1.get(), pv2.get());
            }
        }
    }
}

} // namespace mirheo
