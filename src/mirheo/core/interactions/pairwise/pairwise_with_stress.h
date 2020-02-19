#pragma once

#include "pairwise.h"
#include "kernels/stress_wrapper.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/interactions/interface.h>

#include <map>

namespace mirheo
{

template<class PairwiseKernel>
class PairwiseInteractionWithStress : public BasePairwiseInteraction
{
public:
    using KernelParams = typename PairwiseKernel::ParamsType;

    PairwiseInteractionWithStress(const MirState *state, const std::string& name, real rc, real stressPeriod,
                                  KernelParams pairParams, long seed = 42424242) :
        BasePairwiseInteraction(state, name, rc),
        stressPeriod_(stressPeriod),
        interactionWithoutStress_(state, name, rc, pairParams, seed),
        interactionWithStress_(state, name + "_withStress", rc, pairParams, seed)
    {}

    ~PairwiseInteractionWithStress() = default;
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        interactionWithoutStress_.setPrerequisites(pv1, pv2, cl1, cl2);
        
        pv1->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (ChannelNames::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
        cl2->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
    }

    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const real t = getState()->currentTime;
        
        if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
        {
            debug("Executing interaction '%s' with stress", getCName());
            
            interactionWithStress_.local(pv1, pv2, cl1, cl2, stream);
            lastStressTime_ = t;
        }
        else
        {
            interactionWithoutStress_.local(pv1, pv2, cl1, cl2, stream);
        }
    }
    
    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const real t = getState()->currentTime;
    
        if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
        {
            debug("Executing interaction '%s' with stress", getCName());

            interactionWithStress_.halo(pv1, pv2, cl1, cl2, stream);
            lastStressTime_ = t;
        }
        else
        {
            interactionWithoutStress_.halo(pv1, pv2, cl1, cl2, stream);
        }
    }

    void setSpecificPair(const std::string& pv1name, const std::string& pv2name, const ParametersWrap::MapParams& mapParams) override
    {
        interactionWithoutStress_.setSpecificPair(pv1name, pv2name, mapParams);
        interactionWithStress_   .setSpecificPair(pv1name, pv2name, mapParams);
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        return interactionWithoutStress_.getInputChannels();
    }
    
    std::vector<InteractionChannel> getOutputChannels() const override
    {
        auto channels = interactionWithoutStress_.getOutputChannels();
        
        auto activePredicateStress = [this]()
        {
            const real t = getState()->currentTime;
            return (lastStressTime_+stressPeriod_ <= t) || (lastStressTime_ == t);
        };

        channels.push_back({ChannelNames::stresses, activePredicateStress});

        return channels;
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        interactionWithoutStress_.checkpoint(comm, path, checkpointId);
        interactionWithStress_   .checkpoint(comm, path, checkpointId);
    }
    
    void restart(MPI_Comm comm, const std::string& path) override
    {
        interactionWithoutStress_.restart(comm, path);
        interactionWithStress_   .restart(comm, path);
    }

    static std::string getTypeName()
    {
        return constructTypeName("PairwiseInteractionWithStress", 1,
                                 PairwiseKernel::getTypeName().c_str());
    }
    
private:
    real stressPeriod_;
    real lastStressTime_ {-1e6};

    PairwiseInteraction<PairwiseKernel> interactionWithoutStress_;
    PairwiseInteraction<PairwiseStressWrapper<PairwiseKernel>> interactionWithStress_;
};

} // namespace mirheo
