#pragma once

#include "impl.h"
#include "kernels/stress_wrapper.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/interactions/interface.h>

#include <map>

template<class PairwiseKernel>
class PairwiseInteractionWithStressImpl : public Interaction
{
public:
    PairwiseInteractionWithStressImpl(const MirState *state, const std::string& name, real rc, real stressPeriod, PairwiseKernel pair) :
        Interaction(state, name, rc),
        stressPeriod(stressPeriod),
        interaction(state, name, rc, pair),
        interactionWithStress(state, name + "_withStress", rc, PairwiseStressWrapper<PairwiseKernel>(pair))
    {}

    ~PairwiseInteractionWithStressImpl() = default;
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        interaction.setPrerequisites(pv1, pv2, cl1, cl2);
        
        pv1->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
        cl2->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
    }

    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const real t = state->currentTime;
        
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
        const real t = state->currentTime;
    
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

    void setSpecificPair(const std::string& pv1name, const std::string& pv2name, PairwiseKernel pair)
    {
        interaction.          setSpecificPair(pv1name, pv2name, pair);
        interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseKernel>(pair));
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        return interaction.getInputChannels();
    }
    
    std::vector<InteractionChannel> getOutputChannels() const override
    {
        auto channels = interaction.getOutputChannels();
        
        auto activePredicateStress = [this]()
        {
            const real t = state->currentTime;
            return (lastStressTime+stressPeriod <= t) || (lastStressTime == t);
        };

        channels.push_back({ChannelNames::stresses, activePredicateStress});

        return channels;
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
    real stressPeriod;
    real lastStressTime{-1e6};

    PairwiseInteractionImpl<PairwiseKernel> interaction;
    PairwiseInteractionImpl<PairwiseStressWrapper<PairwiseKernel>> interactionWithStress;
};
