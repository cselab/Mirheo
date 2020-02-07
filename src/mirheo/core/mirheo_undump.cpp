#include "mirheo.h"

#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/interactions/pairwise.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

namespace {
    struct ExtraContext {
        std::map<std::string, std::shared_ptr<ParticleVector>> pvs;
        std::map<std::string, std::shared_ptr<Interaction>> interactions;
    };
} // anonymous namespace

template <typename T>
static const std::shared_ptr<T>&
resolveReference(const std::map<std::string, std::shared_ptr<T>> &objects,
                 const Config& reference)
{
    // Assumed format: "<TYPENAME with name=NAME>".

    const std::string& ref = reference.getString();
    size_t pos = ref.find("with name=");
    if (pos == std::string::npos)
        die("Unrecognized reference format: %s", ref.c_str());
    pos += 4 + 1 + 5;
    std::string name = ref.substr(pos, ref.size() - pos - 1);
    auto it = objects.find(name);
    if (it == objects.end())
        die("Container %p has no object named \"%s\".", (void *)&objects, name.c_str());
    return it->second;
}

static std::shared_ptr<ParticleVector>
importParticleVector(const MirState *state, Undumper& un, const Config::Dictionary& dict)
{
    const std::string& type = dict.at("__type").getString();
    if (type == "ParticleVector") {
        return std::make_shared<ParticleVector>(state, un, dict);
    }

    die("Unknown or unimplemented particle vector type: %s",
        Config{dict}.toJSONString().c_str());
}

static std::shared_ptr<Interaction>
importInteraction [[maybe_unused]] (const MirState *state, Undumper& un, const Config::Dictionary& dict)
{
    const std::string& type = dict.at("__type").getString();
    if (type == "PairwiseInteraction") {
        return std::make_shared<PairwiseInteraction>(state, un, dict);
    }

    die("Unknown or unimplemented interaction type: %s",
        Config{dict}.toJSONString().c_str());
}

void Mirheo::importSnapshot(Undumper& un, const Config& compConfig, const Config& postConfig)
{
    (void)postConfig;

    ExtraContext context;
    std::shared_ptr<InitialConditions> ic = std::make_shared<RestartIC>(un.getContext().path);

    auto importObject = [this, &un](
            const Config& info,
            auto &objects,
            auto func) -> decltype(auto)
    {
        auto it = objects.emplace(
                info.at("name").getString(),
                func(getState(), un, info.getDict())).first;
        debug("Imported object \"%s\" to container %p.\n",
              it->first.c_str(), (void *)&objects);
        return it->second;
    };

    const auto &pvInfos = compConfig.at("ParticleVector").getList();
    auto &pvs = context.pvs;
    for (const auto& info : pvInfos) {
        const auto &pv = importObject(info, pvs, importParticleVector);
        registerParticleVector(pv, ic);
    }

    if (isComputeTask()) {
        const auto &interactionInfos = compConfig.at("Interaction").getList();
        auto &interactions = context.interactions;
        for (const auto& info : interactionInfos) {
            const auto& interaction = importObject(info, interactions, importInteraction);
            registerInteraction(interaction);
        }

        const Config& sim = compConfig.at("Simulation").at(0);
        for (const auto& info : sim.at("interactionPrototypes").getList()) {
            const auto& pv1 = resolveReference(pvs, info.at("pv1"));
            const auto& pv2 = resolveReference(pvs, info.at("pv2"));
            const auto& interaction = resolveReference(interactions, info.at("interaction"));
            real rc = un.undump<real>(info.at("rc"));
            if (rc != interaction->rc)
                die("Interaction rc do not match: %f != %f.", rc, interaction->rc);
            setInteraction(interaction.get(), pv1.get(), pv2.get());
        }
    }
}


} // namespace mirheo
