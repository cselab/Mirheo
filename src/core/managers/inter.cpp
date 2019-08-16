#include "inter.h"

#include <core/celllist.h>

#include <set>

namespace NewInterface
{

void InteractionManager::add(Interaction *interaction,
                             ParticleVector *pv1, ParticleVector *pv2,
                             CellList *cl1, CellList *cl2)
{
    
    interactions.push_back({interaction, pv1, pv2, cl1, cl2});
}

CellList* InteractionManager::getLargestCellList(ParticleVector *pv) const
{
    const auto it = cellListMap.find(pv);
    if (it == cellListMap.end())
        die("pv not found in map: %s", pv->name.c_str());

    auto& cellLists = it->second;

    CellList *clMax {nullptr};
    
    for (auto cl : cellLists)
    {
        if (outputChannels.find(cl) == outputChannels.end())
            continue;
        
        if (clMax == nullptr || cl->rc > clMax->rc)
            clMax = cl;
    }
    
    return clMax;
}

std::vector<std::string> InteractionManager::getInputChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, inputChannels);
}

std::vector<std::string> InteractionManager::getOutputChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, outputChannels);
}



void InteractionManager::clearOutput(ParticleVector *pv, cudaStream_t stream)
{
    auto clListIt = cellListMap.find(pv);

    if (clListIt == cellListMap.end())
        return;

    for (auto cl : clListIt->second)
    {
        auto it = outputChannels.find(cl);
        
        if (it != outputChannels.end()) {
            auto activeChannels = _getActiveChannels(it->second);
            cl->clearChannels(activeChannels, stream);
        }
    }
}

void InteractionManager::accumulateOutput(cudaStream_t stream)
{
    for (const auto& entry : outputChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _getActiveChannels(entry.second);
        cl->accumulateChannels(activeChannels, stream);
    }    
}

void InteractionManager::gatherInputToCells(cudaStream_t stream)
{
    for (const auto& entry : inputChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _getActiveChannels(entry.second);
        cl->gatherChannels(activeChannels, stream);
    }
}

void InteractionManager::executeLocal(cudaStream_t stream)
{
    for (auto& p : interactions)
        p.interaction->local(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}

void InteractionManager::executeHalo (cudaStream_t stream)
{
    for (auto& p : interactions)
        p.interaction->halo(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}


std::vector<std::string> InteractionManager::_getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelList>& allChannels) const
{
    std::set<std::string> extraChannels;

    const auto& clList = cellListMap.find(pv);

    if (clList == cellListMap.end())
        return {};

    for (const auto& cl : clList->second)
    {
        const auto& it = allChannels.find(cl);        
        if (it == allChannels.end())
            continue;

        for (const auto& entry : it->second)
            extraChannels.insert(entry.name);
    }
    return {extraChannels.begin(), extraChannels.end()};

}

std::vector<std::string> InteractionManager::_getActiveChannels(const ChannelList& channelList) const
{
    std::vector<std::string> activeChannels;

    for (const auto& channel : channelList)
        if (channel.active())
            activeChannels.push_back(channel.name);
    
    return activeChannels;
}

} // namespace NewInterface
