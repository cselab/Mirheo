#include "snapshot.h"
#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/integrators/factory.h>
#include <mirheo/core/interactions/factory.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/mesh/factory.h>
#include <mirheo/core/walls/factory.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/pvs/factory.h>
#include <mirheo/core/utils/compile_options.h>
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
    LoaderContext{configFromJSONFile(joinPaths(path, "config.json")),
                  std::move(path)}
{}

LoaderContext::LoaderContext(ConfigValue config, std::string snapshotPath) :
    path_{std::move(snapshotPath)},
    config_{std::move(config)}
{}

LoaderContext::~LoaderContext() = default;


template <typename T, typename Factory>
const std::shared_ptr<T>& loadObject(Mirheo *mir, Loader& loader,
                                     const ConfigValue& info, Factory factory)
{
    auto ptr = factory(mir->getState(), loader, info.getObject());
    if (!ptr)
        die("Factory returned a nullptr for object: %s", info.toJSONString().c_str());
    auto it = loader.getContext().getContainer<T>().emplace(info["name"], std::move(ptr)).first;
    debug("Loaded the object \"%s\".\n", it->first.c_str());
    return it->second;
}


static void loadPlugins(Mirheo *mir, Loader& loader)
{
    auto& context = loader.getContext();
    const ConfigObject& config = context.getConfig();
    const ConfigArray& refsSim  = config["Simulation"][0]["plugins"].getArray();
    const ConfigArray& refsPost = config["Postprocess"][0]["plugins"].getArray();

    // Map from plugin name to the corresponding ConfigObject.
    std::map<std::string, const ConfigObject*> infosSim;
    std::map<std::string, const ConfigObject*> infosPost;
    if (auto *infos = config.get("SimulationPlugin"))
        for (const ConfigValue& info : infos->getArray())
            infosSim.emplace(info["name"], &info.getObject());
    if (auto *infos = config.get("PostprocessPlugin"))
        for (const ConfigValue& info : infos->getArray())
            infosPost.emplace(info["name"], &info.getObject());

    /// Load i-th simulation plugin and j-th postprocess plugin. If the plugin has no
    /// simulation or postprocess plugin object, the corresponding value is -1.
    auto loadPlugin = [&](int i, int j)
    {
        const ConfigObject* configSim =
                i >= 0 ? infosSim.at(parseNameFromRefString(refsSim[i])) : nullptr;
        const ConfigObject* configPost =
                j >= 0 ? infosPost.at(parseNameFromRefString(refsPost[j])) : nullptr;

        const auto &factories = PluginFactoryContainer::get().getFactories();
        for (const auto& factory : factories) {
            auto plugins = factory(mir->isComputeTask(), mir->getState(),
                                   loader, configSim, configPost);
            if (std::get<0>(plugins)) {
                mir->registerPlugins(std::move(std::get<1>(plugins)),
                                     std::move(std::get<2>(plugins)));
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
    while (i < (int)refsSim.size() && j < (int)refsPost.size()) {
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
    // Load remaining simulation and postprocess plugins.
    for (; i < (int)refsSim.size(); ++i)
        loadPlugin(i, -1);
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

/// Check if the snasphot binary format matches the current compilation options.
static void checkCompilationOptions(const ConfigObject& options)
{
    bool useDouble = options["useDouble"];
    if (useDouble != CompileOptions::useDouble) {
        die("Mismatch in the compilation option useDouble: compiled=%d, snapshot=%d",
            (int)useDouble, (int)CompileOptions::useDouble); \
    }
}

/// Load all objects that are defined only on compute ranks.
static void loadComputeSpecificObjects(Mirheo *mir, Loader& loader, const ConfigObject& config)
{
    LoaderContext& context = loader.getContext();
    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(context.getPath());

    if (auto *infos = config.get("Mesh")) {
        for (const auto& info : infos->getArray())
            loadObject<Mesh>(mir, loader, info, loadMesh);
    }

    if (auto *infos = config.get("ParticleVector")) {
        for (const auto& info : infos->getArray()) {
            const auto &pv = loadObject<ParticleVector>(
                    mir, loader, info, particle_vector_factory::loadParticleVector);
            mir->registerParticleVector(pv, ic);
        }
    }

    if (auto *infos = config.get("Wall"))
        for (const auto& info : infos->getArray())
            loadObject<Wall>(mir, loader, info, wall_factory::loadWall);

    if (auto *infos = config.get("Interaction")) {
        for (const auto& info : infos->getArray()) {
            const auto& interaction = loadObject<Interaction>(
                    mir, loader, info, interaction_factory::loadInteraction);
            mir->registerInteraction(interaction);
        }
    }

    if (auto *infos = config.get("Integrator")) {
        for (const auto& info : infos->getArray()) {
            const auto& integrator = loadObject<Integrator>(
                    mir, loader, info, integrator_factory::loadIntegrator);
            mir->registerIntegrator(integrator);
        }
    }

    const ConfigObject& sim = config["Simulation"][0].getObject();
    if (auto *infos = sim.get("interactionPrototypes")) {
        for (const auto& info : infos->getArray()) {
            const auto& pv1 = context.get<ParticleVector>(info["pv1"]);
            const auto& pv2 = context.get<ParticleVector>(info["pv2"]);
            const auto& interaction = context.get<Interaction>(info["interaction"]);
            real rc = loader.load<real>(info["rc"]);
            real irc = interaction->getCutoffRadius();
            if (rc != irc)
                die("Interaction rc do not match: %f != %f.", rc, irc);
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

    if (auto *infos = sim.get("checkWallPrototypes")) {
        for (const auto& info : infos->getArray()) {
            const auto& wall = context.get<Wall>(info["wall"]);
            mir->registerWall(wall, info["every"]);
        }
    }
}

void loadSnapshot(Mirheo *mir, Loader& loader)
{
    // Ensure compute rank first accesses the GPU (on some machines, only one
    // process can access the GPU).
    if (mir->isComputeTask())
        DeviceBuffer<char> tmp(1);
    MPI_Barrier(mir->getWorldComm());

    LoaderContext& context = loader.getContext();
    const ConfigObject& config = context.getConfig();
    const ConfigObject& mirConfig = config["Mirheo"][0].getObject();

    checkCompilationOptions(mirConfig["compile_options"].getObject());

    if (auto* attrs = mirConfig.get("attributes")) {
        for (const auto& pair : attrs->getObject())
            mir->setAttribute(pair.first, pair.second);
    }

    if (mir->isComputeTask())
        loadComputeSpecificObjects(mir, loader, config);

    loadPlugins(mir, loader);
}

} // namespace mirheo
