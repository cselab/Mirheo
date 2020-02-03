#pragma once

#include <mirheo/core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

namespace mirheo
{

class LocalParticleVector;

class InteractionManager
{
public:
    InteractionManager() = default;
    ~InteractionManager() = default;

    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    bool empty() const;
    
    CellList* getLargestCellList(ParticleVector *pv) const;
    real getLargestCutoff() const;

    std::vector<std::string> getInputChannels(ParticleVector *pv) const;
    std::vector<std::string> getOutputChannels(ParticleVector *pv) const;

    void clearInput(ParticleVector *pv, cudaStream_t stream);
    void clearInputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const;

    void clearOutput(ParticleVector *pv, cudaStream_t stream);
    void clearOutputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const;

    void accumulateOutput  (cudaStream_t stream);
    void gatherInputToCells(cudaStream_t stream);

    void executeLocal(cudaStream_t stream);
    void executeHalo (cudaStream_t stream);

    void checkCompatibleWith(const InteractionManager& next) const;
    
private:

    using Channel = Interaction::InteractionChannel;
    
    struct InteractionPrototype
    {
        Interaction *interaction;
        ParticleVector *pv1, *pv2;
        CellList *cl1, *cl2;
    };

    using ChannelList = std::vector<Channel>;

    std::vector<InteractionPrototype> interactions_;
    std::map<CellList*, ChannelList> inputChannels_, outputChannels_;
    std::map<ParticleVector*, std::vector<CellList*>> cellListMap_;

private:

    std::vector<std::string> _getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelList>& allChannels) const;
    std::vector<std::string> _getActiveChannels(const ChannelList& channelList) const;
    std::vector<std::string> _getActiveChannelsFrom(ParticleVector *pv, const std::map<CellList*, ChannelList>& srcChannels) const;
};

} // namespace mirheo
