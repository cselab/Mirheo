#include "interactions.h"

#include <core/celllist.h>

#include <set>

static void insertClist(CellList *cl, std::vector<CellList*>& clists)
{
    auto it = std::find(clists.begin(), clists.end(), cl);

    if (it == clists.end())
        clists.push_back(cl);
}

void InteractionManager::add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    auto intermediateOutput = interaction->getIntermediateOutputChannels();
    auto intermediateInput  = interaction->getIntermediateInputChannels();
    auto finalOutput        = interaction->getFinalOutputChannels();

    if (!intermediateOutput.empty() && !finalOutput.empty())
        die("Interaction '%s' must not have both intermediate and final outputs", interaction->name.c_str());

    if (!intermediateOutput.empty() && !intermediateInput.empty())
        die("Interaction '%s' must not have both intermediate inputs and outputs", interaction->name.c_str());

    if (intermediateOutput.empty() && finalOutput.empty())
        die("Interaction '%s' has no output at all", interaction->name.c_str());
    
    auto addChannels = [&](CellList *cl) {
                           _addChannels(intermediateOutput, cellIntermediateOutputChannels, cl);
                           _addChannels(intermediateInput,  cellIntermediateInputChannels, cl);
                           _addChannels(finalOutput, cellFinalChannels, cl);
                       };

    addChannels(cl1);
    if (cl1 != cl2)
        addChannels(cl2);

    insertClist(cl1, cellListMap[pv1]);
    insertClist(cl2, cellListMap[pv2]);

    InteractionPrototype prototype {interaction, pv1, pv2, cl1, cl2};
    
    if (!intermediateOutput.empty())
        intermediateInteractions.push_back(prototype);

    if (!finalOutput.empty())
        finalInteractions.push_back(prototype);
}

void InteractionManager::check() const
{
    auto outputs = _extractAllChannels(cellIntermediateOutputChannels);
    auto  inputs = _extractAllChannels( cellIntermediateInputChannels);
    std::vector<std::string> difference;

    std::set_difference(inputs.begin(), inputs.end(),
                        outputs.begin(), outputs.end(),
                        std::inserter(difference, difference.begin()));

    if (!difference.empty())
    {
        std::string allChannels;
        for (const auto& ch : difference)
            allChannels += " " + ch;
        die("The following channels are required but not computed by interactions: %s", allChannels.c_str());
    }
}

float InteractionManager::getMaxEffectiveCutoff() const
{
    float rcIntermediate = _getMaxCutoff(cellIntermediateOutputChannels);
    float rcFinal        = _getMaxCutoff(cellFinalChannels);
    return rcIntermediate + rcFinal;
}

CellList* InteractionManager::getLargestCellListNeededForIntermediate(ParticleVector *pv) const
{
    return _getLargestCellListNeeded(pv, cellIntermediateOutputChannels);
}

CellList* InteractionManager::getLargestCellListNeededForFinal(ParticleVector *pv) const
{
    return _getLargestCellListNeeded(pv, cellFinalChannels);
}

std::vector<std::string> InteractionManager::getExtraIntermediateChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, cellIntermediateOutputChannels);
}

std::vector<std::string> InteractionManager::getExtraFinalChannels(ParticleVector *pv) const
{
    return _getExtraChannels(pv, cellFinalChannels);
}


void InteractionManager::clearIntermediates(ParticleVector *pv, cudaStream_t stream)
{
    _clearChannels(pv, cellIntermediateOutputChannels, stream);
    _clearChannels(pv, cellIntermediateInputChannels, stream);
}

void InteractionManager::clearFinal(ParticleVector *pv, cudaStream_t stream)
{
    _clearChannels(pv, cellFinalChannels, stream);
}

void InteractionManager::accumulateIntermediates(cudaStream_t stream)
{
    _accumulateChannels(cellIntermediateOutputChannels, stream);
}

void InteractionManager::accumulateFinal(cudaStream_t stream)
{
    _accumulateChannels(cellFinalChannels, stream);
}

void InteractionManager::gatherIntermediate(cudaStream_t stream)
{
    _gatherChannels(cellIntermediateInputChannels, stream);
}


void InteractionManager::executeLocalIntermediate(cudaStream_t stream)
{
    _executeLocal(intermediateInteractions, stream);
}

void InteractionManager::executeLocalFinal(cudaStream_t stream)
{
    _executeLocal(finalInteractions, stream);
}

void InteractionManager::executeHaloIntermediate(cudaStream_t stream)
{
    _executeHalo(intermediateInteractions, stream);
}

void InteractionManager::executeHaloFinal(cudaStream_t stream)
{
    _executeHalo(finalInteractions, stream);
}


static Interaction::ActivePredicate predicateOr(Interaction::ActivePredicate p1, Interaction::ActivePredicate p2)
{
    return [p1, p2]() {return p1() || p2();};
}

void InteractionManager::_addChannels(const std::vector<Interaction::InteractionChannel>& channels,
                                      std::map<CellList*, ChannelActivityList>& dst,
                                      CellList* cl) const
{
    if (channels.empty()) return;
    
    for (const auto& srcEntry : channels)
    {
        auto it = dst[cl].begin();

        for (; it != dst[cl].end(); ++it)
            if (it->first == srcEntry.name)
                break;

        if (it != dst[cl].end())
            it->second = predicateOr(it->second, srcEntry.active);
        else
            dst[cl].push_back( { srcEntry.name, srcEntry.active } );
    }
}

float InteractionManager::_getMaxCutoff(const std::map<CellList*, ChannelActivityList>& cellChannels) const
{
    float rc = 0.f;
    for (const auto& entry : cellChannels) {
        if (entry.second.empty())
            continue;
        rc = std::max(rc, entry.first->rc);
    }
    return rc;
}

CellList* InteractionManager::_getLargestCellListNeeded(ParticleVector *pv, const std::map<CellList*, ChannelActivityList>& cellChannels) const
{
    CellList *clMax = nullptr;
    
    auto clList = cellListMap.find(pv);

    if (clList == cellListMap.end())
        return nullptr;
    
    for (const auto& cl : clList->second)
    {
        if (cellChannels.find(cl) != cellChannels.end())
        {
            if (clMax == nullptr || cl->rc > clMax->rc)
                clMax = cl;
        }
    }

    return clMax;
}

std::vector<std::string> InteractionManager::_extractAllChannels(const std::map<CellList*, ChannelActivityList>& cellChannels) const
{
    std::set<std::string> channels;
    for (const auto& cellMap : cellChannels)
        for (const auto& entry : cellMap.second)
        {
            std::string name = entry.first;
            channels.insert(name);
        }
    return {channels.begin(), channels.end()};
}

std::vector<std::string> InteractionManager::_getExtraChannels(ParticleVector *pv, const std::map<CellList*, ChannelActivityList>& cellChannels) const
{
    std::set<std::string> channels;

    auto clList = cellListMap.find(pv);

    if (clList == cellListMap.end())
        return {};

    for (const auto& cl : clList->second) {

        auto it = cellChannels.find(cl);        
        if (it == cellChannels.end())
            continue;

        for (const auto& entry : it->second) {
            const std::string& name = entry.first;
            channels.insert(name);
        }
    }
    return {channels.begin(), channels.end()};
}


void InteractionManager::_executeLocal(std::vector<InteractionPrototype>& interactions, cudaStream_t stream)
{
    for (auto& p : interactions)
        p.interaction->local(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}

void InteractionManager::_executeHalo(std::vector<InteractionPrototype>& interactions, cudaStream_t stream)
{
    for (auto& p : interactions)
        p.interaction->halo(p.pv1, p.pv2, p.cl1, p.cl2, stream);
}

std::vector<std::string> InteractionManager::_extractActiveChannels(const ChannelActivityList& activityMap) const
{
    std::vector<std::string> activeChannels;

    for (auto& entry : activityMap)
        if (entry.second())
            activeChannels.push_back(entry.first);
    
    return activeChannels;
}

void InteractionManager::_clearChannels(ParticleVector *pv, const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const
{
    auto clList = cellListMap.find(pv);

    if (clList == cellListMap.end())
        return;

    for (auto cl : clList->second)
    {
        auto it = cellChannels.find(cl);
        
        if (it != cellChannels.end()) {
            auto activeChannels = _extractActiveChannels(it->second);
            cl->clearChannels(activeChannels, stream);
        }
    }
}

void InteractionManager::_accumulateChannels(const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const
{
    for (const auto& entry : cellChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _extractActiveChannels(entry.second);
        cl->accumulateChannels(activeChannels, stream);
    }    
}

void InteractionManager::_gatherChannels(const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const
{
    for (const auto& entry : cellChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _extractActiveChannels(entry.second);
        cl->gatherChannels(activeChannels, stream);
    }
}
