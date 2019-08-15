#pragma once

#include "interface.h"
#include "pairwise.impl.h"
#include "pairwise/interactions/stress_wrapper.h"

#include <core/datatypes.h>
#include <map>

template<class PairwiseInteraction>
class InteractionPair_withStress : public Interaction
{
public:
    enum class InteractionType { Regular, Halo };

    InteractionPair_withStress(const MirState *state, std::string name, float rc, float stressPeriod, PairwiseInteraction pair) :
        Interaction(state, name, rc),
        stressPeriod(stressPeriod),
        interaction(state, name, rc, pair),
        interactionWithStress(state, name + "_withStress", rc, PairwiseStressWrapper<PairwiseInteraction>(pair))
    {}

    ~InteractionPair_withStress() = default;
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        info("Interaction '%s' requires channel '%s' from PVs '%s' and '%s'",
              name.c_str(), ChannelNames::stresses.c_str(), pv1->name.c_str(), pv2->name.c_str());

        pv1->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
        cl2->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
    }

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        float t = state->currentTime;
        
        if (lastStressTime+stressPeriod <= t || lastStressTime == t)
        {
            debug("Executing interaction '%s' with stress", name.c_str());
            
            interactionWithStress.local(pv1, pv2, cl1, cl2, stream);
            lastStressTime = t;
        }
        else
        {
            interaction.local(pv1, pv2, cl1, cl2, stream);
        }
    }
    
    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        float t = state->currentTime;
    
        if (lastStressTime+stressPeriod <= t || lastStressTime == t)
        {
            debug("Executing interaction '%s' with stress", name.c_str());

            interactionWithStress.halo(pv1, pv2, cl1, cl2, stream);
            lastStressTime = t;
        }
        else
        {
            interaction.halo(pv1, pv2, cl1, cl2, stream);
        }
    }

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair)
    {
        interaction.          setSpecificPair(pv1name, pv2name, pair);
        interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(pair));
    }

    std::vector<InteractionChannel> getFinalOutputChannels() const override
    {
        auto activePredicateStress = [this]() {
            float t = state->currentTime;
            return (lastStressTime+stressPeriod <= t) || (lastStressTime == t);
        };

        return {{ChannelNames::forces, Interaction::alwaysActive},
                {ChannelNames::stresses, activePredicateStress}};
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        interaction          .checkpoint(comm, path, checkpointId);
        interactionWithStress.checkpoint(comm, path, checkpointId);
    }
    
    void restart(MPI_Comm comm, const std::string& path) override
    {
        interaction          .restart(comm, path);
        interactionWithStress.restart(comm, path);
    }

    
private:
    float stressPeriod;
    float lastStressTime{-1e6};

    InteractionPair<PairwiseInteraction> interaction;
    InteractionPair<PairwiseStressWrapper<PairwiseInteraction>> interactionWithStress;
};
