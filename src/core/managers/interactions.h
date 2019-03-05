#pragma once

#include <core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

class ParticleVector;
class CellList;

/**
 * Interaction manager.
 *
 * There are two kinds of interactions:
 * - 'Final' interactions, responsible to output final quantities, e.g.  forces, stresses
 * - 'Intermediate' interactions, which compute intermediate quantities required by final interactions, e.g. particle densities
 *
 * This class is a managing clearing, gathering and accumulating the channels of the different cell lists.
 * It also wraps the execution of the interactions 
 */
class InteractionManager
{
public:
    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);
    void check() const;

    float getMaxEffectiveCutoff() const;
    
    CellList* getLargestCellListNeededForIntermediate(const std::vector<std::unique_ptr<CellList>>& cellListVec) const;
    CellList* getLargestCellListNeededForFinal       (const std::vector<std::unique_ptr<CellList>>& cellListVec) const;

    std::vector<std::string> getExtraIntermediateChannels(const std::vector<std::unique_ptr<CellList>>& cellListVec) const;
    std::vector<std::string> getExtraFinalChannels(const std::vector<std::unique_ptr<CellList>>& cellListVec) const;    
    
    void clearIntermediates (CellList *cl, cudaStream_t stream);
    void clearFinal         (CellList *cl, cudaStream_t stream);

    void accumulateIntermediates(cudaStream_t stream);
    void accumulateFinal(cudaStream_t stream);

    void gatherIntermediate(cudaStream_t stream);

    void executeLocalIntermediate(cudaStream_t stream);
    void executeLocalFinal(cudaStream_t stream);

    void executeHaloIntermediate(cudaStream_t stream);
    void executeHaloFinal(cudaStream_t stream);

private:

    using ChannelActivityList = std::vector<std::pair<std::string, Interaction::ActivePredicate>>;
    
    std::map<CellList*, ChannelActivityList> cellIntermediateOutputChannels;
    std::map<CellList*, ChannelActivityList> cellIntermediateInputChannels;
    std::map<CellList*, ChannelActivityList> cellFinalChannels;
    std::map<ParticleVector*, std::vector<CellList*>> cellListMap;
    
    struct InteractionPrototype
    {
        Interaction *interaction;
        ParticleVector *pv1, *pv2;
        CellList *cl1, *cl2;
    };

    std::vector<InteractionPrototype> intermediateInteractions;
    std::vector<InteractionPrototype> finalInteractions;

private:

    void _addChannels(const std::vector<Interaction::InteractionChannel>& channels,
    				  std::map<CellList*, ChannelActivityList>& dst,
                      CellList* cl) const;

    float _getMaxCutoff(const std::map<CellList*, ChannelActivityList>& cellChannels) const;
    
    CellList* _getLargestCellListNeeded(const std::map<CellList*, ChannelActivityList>& cellChannels,
                                        const std::vector<std::unique_ptr<CellList>>& cellListVec) const;

    std::vector<std::string> _getExtraChannels(const std::map<CellList*, ChannelActivityList>& cellChannels,
                                               const std::vector<std::unique_ptr<CellList>>& cellListVec) const;
    
    void _executeLocal(std::vector<InteractionPrototype>& interactions, cudaStream_t stream);
    void _executeHalo(std::vector<InteractionPrototype>& interactions, cudaStream_t stream);

    std::vector<std::string> _extractActiveChannels(const ChannelActivityList& activityMap) const;
    std::vector<std::string> _extractAllChannels(const std::map<CellList*, ChannelActivityList>& cellChannels) const;
    
    void _clearChannels     (CellList *cl, const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const;
    void _accumulateChannels(const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const;
    void _gatherChannels    (const std::map<CellList*, ChannelActivityList>& cellChannels, cudaStream_t stream) const;
};
