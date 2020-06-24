// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interactions.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/particle_vector.h>

#include <algorithm>
#include <set>

namespace mirheo
{

static inline Interaction::ActivePredicate predicateOr(Interaction::ActivePredicate p1, Interaction::ActivePredicate p2)
{
    return [p1, p2]() {return p1() || p2();};
}

static void insertClist(CellList *cl, std::vector<CellList*>& clists)
{
    auto it = std::find(clists.begin(), clists.end(), cl);

    if (it == clists.end())
        clists.push_back(cl);
}

void InteractionManager::add(Interaction *interaction,
                             ParticleVector *pv1, ParticleVector *pv2,
                             CellList *cl1, CellList *cl2)
{
    const auto input  = interaction->getInputChannels();
    const auto output = interaction->getOutputChannels();

    auto insertChannels = [&](CellList *cl)
    {
        auto _insertChannels = [&](const std::vector<Interaction::InteractionChannel>& channels,
                                   std::map<CellList*, ChannelList>& dst)
        {
            if (channels.empty())
                return; // otherwise will create an empty entry at cl because of operator[]

            for (const auto& srcEntry : channels)
            {
                auto it = dst[cl].begin();

                for (; it != dst[cl].end(); ++it)
                    if (it->name == srcEntry.name)
                        break;

                if (it != dst[cl].end())
                    it->active = predicateOr(it->active, srcEntry.active);
                else
                    dst[cl].push_back( srcEntry );
            }
        };

        _insertChannels( input,  inputChannels_);
        _insertChannels(output, outputChannels_);
    };

    insertChannels(cl1);
    if (cl1 != cl2)
        insertChannels(cl2);

    insertClist(cl1, cellListMap_[pv1]);
    insertClist(cl2, cellListMap_[pv2]);

    interactions_.push_back({interaction, pv1, pv2, cl1, cl2});
}

bool InteractionManager::empty() const
{
    return interactions_.empty();
}

CellList* InteractionManager::getLargestCellList(ParticleVector *pv) const
{
    const auto it = cellListMap_.find(pv);
    if (it == cellListMap_.end())
        return nullptr;

    auto& cellLists = it->second;

    CellList *clMax {nullptr};

    for (auto cl : cellLists)
    {
        if (outputChannels_.find(cl) == outputChannels_.end())
            continue;

        if (clMax == nullptr || cl->rc > clMax->rc)
            clMax = cl;
    }

    return clMax;
}

real InteractionManager::getLargestCutoff() const
{
    real rc = 0.0_r;
    for (const auto& prototype : interactions_)
    {
        if (!prototype.cl1)
            continue;
        rc = std::max(rc, prototype.cl1->rc);
    }
    return rc;
}

std::vector<std::string> InteractionManager::getInputChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, inputChannels_);
}

std::vector<std::string> InteractionManager::getOutputChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, outputChannels_);
}

void InteractionManager::clearInput(ParticleVector *pv, cudaStream_t stream)
{
    auto clListIt = cellListMap_.find(pv);

    if (clListIt == cellListMap_.end())
        return;

    for (auto cl : clListIt->second)
    {
        auto it = inputChannels_.find(cl);

        if (it != inputChannels_.end()) {
            const auto activeChannels = _getActiveChannels(it->second);
            cl->clearChannels(activeChannels, stream);
        }
    }
}

void InteractionManager::clearInputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const
{
    const auto activeChannels = _getActiveChannelsFrom(pv, inputChannels_);

    for (const auto& channelName : activeChannels)
        lpv->dataPerParticle.getGenericData(channelName)->clearDevice(stream);
}


void InteractionManager::clearOutput(ParticleVector *pv, cudaStream_t stream)
{
    auto clListIt = cellListMap_.find(pv);

    if (clListIt == cellListMap_.end())
        return;

    for (auto cl : clListIt->second)
    {
        auto it = outputChannels_.find(cl);

        if (it != outputChannels_.end()) {
            auto activeChannels = _getActiveChannels(it->second);
            cl->clearChannels(activeChannels, stream);
        }
    }
}

void InteractionManager::clearOutputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const
{
    const auto activeChannels = _getActiveChannelsFrom(pv, outputChannels_);

    for (const auto& channelName : activeChannels)
        lpv->dataPerParticle.getGenericData(channelName)->clearDevice(stream);
}

void InteractionManager::accumulateOutput(cudaStream_t stream)
{
    for (const auto& entry : outputChannels_)
    {
        auto cl = entry.first;
        auto activeChannels = _getActiveChannels(entry.second);
        cl->accumulateChannels(activeChannels, stream);
    }
}

void InteractionManager::gatherInputToCells(cudaStream_t stream)
{
    for (const auto& entry : inputChannels_)
    {
        auto cl = entry.first;
        auto activeChannels = _getActiveChannels(entry.second);
        cl->gatherChannels(activeChannels, stream);
    }
}

void InteractionManager::executeLocal(cudaStream_t stream)
{
    for (auto& p : interactions_)
        p.interaction->local(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}

void InteractionManager::executeHalo (cudaStream_t stream)
{
    for (auto& p : interactions_)
        p.interaction->halo(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}


static std::set<std::string> getAllChannelsNames(const std::map<CellList*, std::vector<Interaction::InteractionChannel>>& cellChannels)
{
    std::set<std::string> channels;
    for (const auto& cellMap : cellChannels)
        for (const auto& entry : cellMap.second)
            channels.insert(entry.name);
    return channels;
}

static std::string concatenate(const std::vector<std::string>& strings)
{
    std::string allNames;
    for (const auto& str : strings)
        allNames += " " + str;
    return allNames;
}

void InteractionManager::checkCompatibleWith(const InteractionManager& next) const
{
    const auto outputs = getAllChannelsNames(this->outputChannels_);
    const auto inputs  = getAllChannelsNames(next.inputChannels_);
    std::vector<std::string> difference;

    std::set_difference(inputs.begin(), inputs.end(),
                        outputs.begin(), outputs.end(),
                        std::inserter(difference, difference.begin()));

    if (!difference.empty())
        die("The following channels are required but not computed by interactions: %s", concatenate(difference).c_str());
}


std::vector<std::string> InteractionManager::_getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelList>& allChannels) const
{
    std::set<std::string> extraChannels;

    const auto& clList = cellListMap_.find(pv);

    if (clList == cellListMap_.end())
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

std::vector<std::string> InteractionManager::_getActiveChannelsFrom(ParticleVector *pv, const std::map<CellList*, ChannelList>& srcChannels) const
{
    auto itCMap = cellListMap_.find(pv);
    if (itCMap == cellListMap_.end())
        return {};

    std::set<std::string> channels;

    for (const auto& cl : itCMap->second)
    {
        auto it = srcChannels.find(cl);
        if (it == srcChannels.end())
            continue;

        for (const auto& entry : it->second)
        {
            if (entry.active())
                channels.insert(entry.name);
        }
    }
    return {channels.begin(), channels.end()};
}

} // namespace mirheo
