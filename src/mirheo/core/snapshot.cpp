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
#include <mirheo/plugins/factory.h>

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

bool SaverContext::isGroupMasterTask() const
{
    int rank;
    MPI_Comm_rank(groupComm, &rank);
    return rank == 0;
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
const std::shared_ptr<T>& loadObject(Mirheo *mir, Loader& loader,
                                     const ConfigValue& info, Factory factory)
{
    auto ptr = factory(mir->getState(), loader, info.getObject(), info["__type"]);
    if (!ptr)
        die("Factory returned a nullptr for object: %s", info.toJSONString().c_str());
    auto it = loader.getContext().getContainer<T>().emplace(info["name"], std::move(ptr)).first;
    debug("Loaded the object \"%s\".\n", it->first.c_str());
    return it->second;
}


static void loadPlugins(Mirheo *mir, Loader& loader)
{
    auto& context = loader.getContext();
    const ConfigValue *_refsSim  = context.getComp()["Simulation"][0].get("plugins");
    const ConfigValue *_refsPost = context.getPost()["Postprocess"][0].get("plugins");

    const ConfigArray empty;
    const ConfigArray& refsSim  = _refsSim ? _refsSim->getArray() : empty;
    const ConfigArray& refsPost = _refsPost ? _refsPost->getArray() : empty;

    // Map from plugin name to the corresponding ConfigObject.
    std::map<std::string, const ConfigObject*> infosSim;
    std::map<std::string, const ConfigObject*> infosPost;
    if (auto *infos = context.getComp().get("SimulationPlugin"))
        for (const ConfigValue& info : infos->getArray())
            infosSim.emplace(info["name"], &info.getObject());
    if (auto *infos = context.getPost().get("PostprocessPlugin"))
        for (const ConfigValue& info : infos->getArray())
            infosPost.emplace(info["name"], &info.getObject());

    /// Load i-th simulation plugin and j-th postprocess plugin. If the plugin has no
    /// simulation or postprocess plugin object, the corresponding value is -1.
    auto loadPlugin = [&](int i, int j)
    {
        const ConfigObject* configSim =
                i >= 0 ? infosSim.at(parseNameFromRefString(refsSim[i])) : nullptr;
        const ConfigObject* configPost =
                j >= 0 ? infosPost.at(parseNameFromRefString(refsPost[i])) : nullptr;

        const auto &factories = PluginFactoryContainer::get().getFactories();
        for (const auto& factory : factories) {
            auto plugins = factory(mir->isComputeTask(), mir->getState(),
                                   loader, configSim, configPost);
            if (plugins.first != nullptr || plugins.second != nullptr) {
                mir->registerPlugins(std::move(plugins));
                return;
            }
        }
        die("None of the %zu factories implements or recognizes a plugin pair (%s, %s).",
            factories.size(),
            configSim  ? configSim ->at("__type").getString().c_str() : "<null>",
            configPost ? configPost->at("__type").getString().c_str() : "<null>");
    };

    // We have to load compute and postprocess plugins in the correct order,
    // while having only (L1) the ordered list of compute plugins and (L2) the
    // ordered list of postprocess plugins. There are three kinds of plugins:
    // (A) compute-only, (B) postprocess-only and (C) combined. The code below
    // iterates simultaneously through L1 and L2 and loads plugins in such a
    // way to preserve the order between two As, between As and CS, and between
    // Bs and Cs. The order between As and Bs is not necessarily preserved.
    int i = 0, j = 0;
    while (i < (int)refsSim.size()) {
        // Look for a postprocess plugin with a matching name.
        std::string nameSim = parseNameFromRefString(refsSim[i]);
        auto it = infosPost.find(nameSim);

        if (it == infosPost.end()) {
            // Matching postprocess plugin not found.
            loadPlugin(i, -1);
            ++i;
        } else if (parseNameFromRefString(refsPost[j]) != nameSim) {
            // Found, but there are other postprocess plugins that have to be loaded first.
            loadPlugin(-1, j);
            ++j;
        } else {
            // Found, load simulation and postprocess plugins together.
            loadPlugin(i, j);
            ++i;
            ++j;
        }
    }
    // Load remaining postprocess plugins.
    for (; j < (int)refsPost.size(); ++j)
        loadPlugin(-1, j);
}

PluginFactoryContainer& PluginFactoryContainer::get() noexcept
{
    static PluginFactoryContainer singleton;
    return singleton;
}

void PluginFactoryContainer::registerPluginFactory(FactoryType factory)
{
    factories_.push_back(std::move(factory));
}

void loadSnapshot(Mirheo *mir, Loader& loader)
{
    LoaderContext& context = loader.getContext();
    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(context.getPath());

    if (auto *infos = context.getComp().get("Mesh")) {
        for (const auto& info : infos->getArray())
            loadObject<Mesh>(mir, loader, info, loadMesh);
    }

    if (auto *infos = context.getComp().get("ParticleVector")) {
        for (const auto& info : infos->getArray()) {
            const auto &pv = loadObject<ParticleVector>(
                    mir, loader, info, ParticleVectorFactory::loadParticleVector);
            mir->registerParticleVector(pv, ic);
        }
    }

    if (mir->isComputeTask()) {
        if (auto *infos = context.getComp().get("Interaction")) {
            for (const auto& info : infos->getArray()) {
                const auto& interaction = loadObject<Interaction>(
                        mir, loader, info, InteractionFactory::loadInteraction);
                mir->registerInteraction(interaction);
            }
        }

        if (auto *infos = context.getComp().get("Integrator")) {
            for (const auto& info : infos->getArray()) {
                const auto& integrator = loadObject<Integrator>(
                        mir, loader, info, IntegratorFactory::loadIntegrator);
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

    loadPlugins(mir, loader);
}

} // namespace mirheo
