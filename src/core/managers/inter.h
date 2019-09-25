#pragma once

#include <core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

class LocalParticleVector;

// namespace NewInterface
// {

class InteractionManager
{
public:
    InteractionManager() = default;
    ~InteractionManager() = default;

    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    CellList* getLargestCellList(ParticleVector *pv) const;
    float getLargestCutoff() const;

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
    
private:

    using Channel = Interaction::InteractionChannel;
    
    struct InteractionPrototype
    {
        Interaction *interaction;
        ParticleVector *pv1, *pv2;
        CellList *cl1, *cl2;
    };

    using ChannelList = std::vector<Channel>;

    std::vector<InteractionPrototype> interactions;
    std::map<CellList*, ChannelList> inputChannels, outputChannels;
    std::map<ParticleVector*, std::vector<CellList*>> cellListMap;

private:

    std::vector<std::string> _getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelList>& allChannels) const;
    std::vector<std::string> _getActiveChannels(const ChannelList& channelList) const;
    std::vector<std::string> _getActiveChannelsFrom(ParticleVector *pv, const std::map<CellList*, ChannelList>& srcChannels) const;
};

// } // namespace NewInterface
