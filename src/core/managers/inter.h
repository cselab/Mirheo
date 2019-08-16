#pragma once

#include <core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

namespace NewInterface
{

class InteractionManager
{
public:
    InteractionManager() = default;
    ~InteractionManager() = default;

    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    CellList* getLargestCellList(ParticleVector *pv) const;

    std::vector<std::string> getInputChannels(ParticleVector *pv) const;
    std::vector<std::string> getOutputChannels(ParticleVector *pv) const;

    void clearOutput(ParticleVector *pv, cudaStream_t stream);

    void accumulateOutput  (cudaStream_t stream);
    void gatherInputToCells(cudaStream_t stream);

    void executeLocal(cudaStream_t stream);
    void executeHalo (cudaStream_t stream);
    
private:

    struct Channel
    {
        std::string name;
        Interaction::ActivePredicate active;
    };

    using ChannelList = std::vector<Channel>;

    std::map<CellList*, ChannelList> inputChannels, outputChannels;
    std::map<ParticleVector*, std::vector<CellList*>> cellListMap;

private:

    std::vector<std::string> _getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelList>& allChannels) const;
    std::vector<std::string> _getActiveChannels(const ChannelList& channelList) const;
};

} // namespace NewInterface
