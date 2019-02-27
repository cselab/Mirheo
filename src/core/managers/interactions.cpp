#include "interactions.h"

#include <core/celllist.h>



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

void InteractionManager::clearIntermediates(cudaStream_t stream)
{
    _clearChannels(cellIntermediateOutputChannels, stream);
    _clearChannels(cellIntermediateInputChannels, stream);
}

void InteractionManager::clearFinal(cudaStream_t stream)
{
    _clearChannels(cellFinalChannels, stream);
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

void InteractionManager::_clearChannels(const std::map<CellList*, ChannelActivityMap>& cellChannels, cudaStream_t stream) const
{
    for (const auto& entry : cellChannels)
    {
        auto cl = entry.first;
        auto activeChannels = _extractActiveChannels(entry.second);
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
