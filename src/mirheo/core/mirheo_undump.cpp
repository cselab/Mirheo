#include "mirheo_undump.h"
#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/interactions/membrane.h>
#include <mirheo/core/interactions/pairwise.h>
#include <mirheo/core/mesh/membrane.h>
#include <mirheo/core/mesh/mesh.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/folders.h>

#include <typeinfo>

namespace mirheo
{

std::string _parseNameFromReference(const std::string &ref)
{
    // Assumed format: "<TYPENAME with name=NAME>".
    size_t pos = ref.find("with name=");
    if (pos == std::string::npos)
        die("Unrecognized reference format: %s", ref.c_str());
    pos += 4 + 1 + 5;
    return ref.substr(pos, ref.size() - pos - 1);
}

void _unknownReferenceError [[noreturn]] (const std::string &name)
{
    die("Unknown object reference \"%s\".", name.c_str());
}


// TODO: Make a mesh factory file.
static std::shared_ptr<Mesh>
importMesh(const MirState *, Undumper& un,
           const ConfigObject& config, const std::string &type)
{
    if (type == "Mesh")
        return std::make_shared<Mesh>(un, config);
    if (type == "MembraneMesh")
        return std::make_shared<MembraneMesh>(un, config);
    return {};
}


// TODO: Make a PV factory file.
static std::shared_ptr<ParticleVector>
importParticleVector(const MirState *state, Undumper& un,
                     const ConfigObject& config, const std::string &type)
{
    if (type == "ParticleVector")
        return std::make_shared<ParticleVector>(state, un, config);
    if (type == "MembraneVector")
        return std::make_shared<MembraneVector>(state, un, config);
    return {};
}


// TODO: Make an interaction factory file.
static std::shared_ptr<Interaction>
importInteraction(const MirState *state, Undumper& un,
                  const ConfigObject& config, const std::string &type)
{
    if (type == "PairwiseInteraction")
        return std::make_shared<PairwiseInteraction>(state, un, config);
    if (type == "MembraneInteraction")
        return std::make_shared<MembraneInteraction>(state, un, config);
    return {};
}

UndumpContext::UndumpContext(std::string path) :
    UndumpContext{configFromJSONFile(joinPaths(path, "config.compute.json")),
                  configFromJSONFile(joinPaths(path, "config.post.json")),
                  std::move(path)}
{}

UndumpContext::UndumpContext(ConfigValue compute, ConfigValue postprocess,
                             std::string snapshotPath) :
    path_{std::move(snapshotPath)},
    compConfig_{std::move(compute)},
    postConfig_{std::move(postprocess)}
{}

UndumpContext::~UndumpContext() = default;

const ConfigObject& UndumpContext::getCompObjectConfig(
        const std::string& category, const std::string& name)
{
    for (const ConfigValue& obj : getComp()[category].getList())
        if (obj.getObject()["name"].getString() == name)
            return obj.getObject();
    die("Object category=\"%s\" name=\"%s\" not found.", category.c_str(), name.c_str());
}

template <typename T, typename U>
static std::shared_ptr<T> dynamic_ptr_cast(const std::shared_ptr<U>& ptr)
{
    if (auto *p = dynamic_cast<T*>(ptr.get()))
        return {ptr, p};
    die("Failed dynamic cast from type \"%s\" to type \"%s\".",
        typeid(U).name(), typeid(T).name());
}

std::shared_ptr<MembraneMesh> UndumpContextGetPtr<MembraneMesh>::getShared(
        UndumpContext *context, const std::string &ref)
{
    return dynamic_ptr_cast<MembraneMesh>(context->getShared<Mesh>(ref));
}


void importSnapshot(Mirheo *mir, Undumper& un)
{
    auto& context = un.getContext();
    const ConfigValue& compConfig = context.getComp();

    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(context.getPath());

    auto importObject = [mir, &un](
            const ConfigValue& info,
            auto &objects,
            auto func) -> decltype(auto)
    {
        auto ptr = func(mir->getState(), un, info.getObject(), info["__type"]);
        if (!ptr)
            die("Unknown or unimplemented object type: %s", info.toJSONString().c_str());
        auto it = objects.emplace(info["name"], std::move(ptr)).first;
        debug("Imported object \"%s\" to container %p.\n",
              it->first.c_str(), (void *)&objects);
        return it->second;
    };

    if (auto *meshInfos = compConfig.get("Mesh")) {
        auto &meshes = context.getContainerShared<Mesh>();
        for (const auto& info : meshInfos->getList())
            importObject(info, meshes, importMesh);
    }

    if (auto *pvInfos = compConfig.get("ParticleVector")) {
        auto &pvs = context.getContainerShared<ParticleVector>();
        for (const auto& info : pvInfos->getList()) {
            const auto &pv = importObject(info, pvs, importParticleVector);
            mir->registerParticleVector(pv, ic);
        }
    }

    if (mir->isComputeTask()) {
        if (auto *interactionInfos = compConfig.get("Interaction")) {
            auto &interactions = context.getContainerShared<Interaction>();
            for (const auto& info : interactionInfos->getList()) {
                const auto& interaction = importObject(info, interactions, importInteraction);
                mir->registerInteraction(interaction);
            }
        }

        const ConfigValue& sim = compConfig["Simulation"][0];
        if (auto *interactionPrototypes = sim.get("interactionPrototypes")) {
            for (const auto& info : interactionPrototypes->getList()) {
                const auto& pv1 = context.getShared<ParticleVector>(info["pv1"]);
                const auto& pv2 = context.getShared<ParticleVector>(info["pv2"]);
                const auto& interaction = context.getShared<Interaction>(info["interaction"]);
                real rc = un.undump<real>(info["rc"]);
                if (rc != interaction->rc)
                    die("Interaction rc do not match: %f != %f.", rc, interaction->rc);
                mir->setInteraction(interaction.get(), pv1.get(), pv2.get());
            }
        }
    }
}

} // namespace mirheo
