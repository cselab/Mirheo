#include "interactions.h"

#include <core/celllist.h>

#include <set>

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
                           _addChannels(intermediateOutput, cellIntermediateOutputChannels[cl]);
                           _addChannels(intermediateInput,  cellIntermediateInputChannels[cl]);
                           _addChannels(finalOutput, cellFinalChannels[cl]);
                       };

    addChannels(cl1);
    if (cl1 != cl2)
        addChannels(cl2);

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

CellList* InteractionManager::getLargestCellListNeededForIntermediate(const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    return _getLargestCellListNeeded(cellIntermediateOutputChannels, cellListVec);
}

CellList* InteractionManager::getLargestCellListNeededForFinal(const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    return _getLargestCellListNeeded(cellFinalChannels, cellListVec);
}

std::vector<std::string> InteractionManager::getExtraIntermediateChannels(const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    return _getExtraChannels(cellIntermediateOutputChannels, cellListVec);
}

std::vector<std::string> InteractionManager::getExtraFinalChannels(const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    return _getExtraChannels(cellFinalChannels, cellListVec);
}


void InteractionManager::clearIntermediates(CellList *cl, cudaStream_t stream)
{
    _clearChannels(cl, cellIntermediateOutputChannels, stream);
    _clearChannels(cl, cellIntermediateInputChannels, stream);
}

void InteractionManager::clearFinal(CellList *cl, cudaStream_t stream)
{
    _clearChannels(cl, cellFinalChannels, stream);
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

void InteractionManager::_addChannels(const std::vector<Interaction::InteractionChannel>& src,
                                      std::map<std::string, Interaction::ActivePredicate>& dst) const
{
    for (const auto& srcEntry : src)
    {
        auto it = dst.find(srcEntry.name);

        if (it != dst.end())
            it->second = predicateOr(it->second, srcEntry.active);
        else
            dst[srcEntry.name] = srcEntry.active;
    }
}

float InteractionManager::_getMaxCutoff(const std::map<CellList*, ChannelActivityMap>& cellChannels) const
{
    float rc = 0.f;
    for (const auto& entry : cellChannels) {
        if (entry.second.empty())
            continue;
        rc = std::max(rc, entry.first->rc);
    }
    return rc;
}

static void checkCellListsAreSorted(const std::vector<std::unique_ptr<CellList>>& cellListVec)
{
    for (int i = 1; i < cellListVec.size(); ++i)
        if (cellListVec[i]->rc > cellListVec[i-1]->rc)
            die("Expected sorted cell lists (with decreasing cutoff radius)");
}

CellList* InteractionManager::_getLargestCellListNeeded(const std::map<CellList*, ChannelActivityMap>& cellChannels,
                                                        const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    checkCellListsAreSorted(cellListVec);
    
    for (const auto& cl : cellListVec)
    {
        auto clPtr = cl.get();
        if (cellChannels.find(clPtr) != cellChannels.end())
            return clPtr;
    }
    return nullptr;
}

std::vector<std::string> InteractionManager::_extractAllChannels(const std::map<CellList*, ChannelActivityMap>& cellChannels) const
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

std::vector<std::string> InteractionManager::_getExtraChannels(const std::map<CellList*, ChannelActivityMap>& cellChannels,
                                                               const std::vector<std::unique_ptr<CellList>>& cellListVec) const
{
    std::set<std::string> channels;
    
    for (const auto& cl : cellListVec)
    {
        auto it = cellChannels.find(cl.get());
        
        if (it != cellChannels.end())
        {
            for (const auto& entry : it->second)
            {
                std::string name = entry.first;
                if (name != ChannelNames::forces)
                    channels.insert(name);
            }
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

std::vector<std::string> InteractionManager::_extractActiveChannels(const ChannelActivityMap& activityMap) const
{
    std::vector<std::string> activeChannels;

    for (auto& entry : activityMap)
        if (entry.second())
            activeChannels.push_back(entry.first);
    
    return activeChannels;
}

void InteractionManager::_clearChannels(CellList *cl, const std::map<CellList*, ChannelActivityMap>& cellChannels, cudaStream_t stream) const
{
    auto it = cellChannels.find(cl);

    if (it != cellChannels.end()) {
        auto activeChannels = _extractActiveChannels(it->second);
        cl->clearChannels(activeChannels, stream);
    }
}

void InteractionManager::_accumulateChannels(const std::map<CellList*, ChannelActivityMap>& cellChannels, cudaStream_t stream) const
{
    for (const auto& entry : cellChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _extractActiveChannels(entry.second);
        cl->accumulateChannels(activeChannels, stream);
    }    
}

void InteractionManager::_gatherChannels(const std::map<CellList*, ChannelActivityMap>& cellChannels, cudaStream_t stream) const
{
    for (const auto& entry : cellChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _extractActiveChannels(entry.second);
        cl->gatherChannels(activeChannels, stream);
    }
}
