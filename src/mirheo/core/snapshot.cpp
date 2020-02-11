#include "snapshot.h"
#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/integrators/factory.h>
#include <mirheo/core/interactions/factory.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/mesh/factory.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/pvs/factory.h>
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

template <typename T, typename Factory>
const std::shared_ptr<T>& loadObject(Loader& loader, Mirheo *mir,
                                     const ConfigValue& info, Factory factory)
{
    auto ptr = factory(mir->getState(), loader, info.getObject(), info["__type"]);
    if (!ptr)
        die("Factory returned a nullptr for object: %s", info.toJSONString().c_str());
    auto it = loader.getContext().getContainer<T>().emplace(info["name"], std::move(ptr)).first;
    debug("Loaded the object \"%s\".\n", it->first.c_str());
    return it->second;
}


void loadSnapshot(Mirheo *mir, Loader& loader)
{
    LoaderContext& context = loader.getContext();
    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(context.getPath());

    if (auto *infos = context.getComp().get("Mesh")) {
        for (const auto& info : infos->getArray())
            loadObject<Mesh>(loader, mir, info, loadMesh);
    }

    if (auto *infos = context.getComp().get("ParticleVector")) {
        for (const auto& info : infos->getArray()) {
            const auto &pv = loadObject<ParticleVector>(
                    loader, mir, info, ParticleVectorFactory::loadParticleVector);
            mir->registerParticleVector(pv, ic);
        }
    }

    if (mir->isComputeTask()) {
        if (auto *infos = context.getComp().get("Interaction")) {
            for (const auto& info : infos->getArray()) {
                const auto& interaction = loadObject<Interaction>(
                        loader, mir, info, InteractionFactory::loadInteraction);
                mir->registerInteraction(interaction);
            }
        }

        if (auto *infos = context.getComp().get("Integrator")) {
            for (const auto& info : infos->getArray()) {
                const auto& integrator = loadObject<Integrator>(
                        loader, mir, info, IntegratorFactory::loadIntegrator);
                mir->registerIntegrator(integrator);
            }
        }

        const ConfigObject& sim = context.getComp()["Simulation"][0].getObject();
        if (auto *infos = sim.get("interactionPrototypes")) {
            for (const auto& info : infos->getArray()) {
                const auto& pv1 = context.get<ParticleVector>(info["pv1"]);
                const auto& pv2 = context.get<ParticleVector>(info["pv2"]);
                const auto& interaction = context.get<Interaction>(info["interaction"]);
                real rc = loader.load<real>(info["rc"]);
                if (rc != interaction->rc)
                    die("Interaction rc do not match: %f != %f.", rc, interaction->rc);
                mir->setInteraction(interaction.get(), pv1.get(), pv2.get());
            }
        }

        if (auto *infos = sim.get("integratorPrototypes")) {
            for (const auto& info : infos->getArray()) {
                const auto& pv = context.get<ParticleVector>(info["pv"]);
                const auto& integrator = context.get<Integrator>(info["integrator"]);
                mir->setIntegrator(integrator.get(), pv.get());
            }
        }
    }
}

} // namespace mirheo
