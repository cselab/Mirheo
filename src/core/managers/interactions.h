#pragma once

#include <core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

class ParticleVector;
class CellList;

class InteractionManager
{
public:
    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    void clearIntermediates(cudaStream_t stream);
    void clearFinal(cudaStream_t stream);

    void accumulateIntermediates(cudaStream_t stream);
    void accumulateFinal(cudaStream_t stream);

    void scatterIntermediate(cudaStream_t stream);

    void executeLocalIntermediate(cudaStream_t stream);
    void executeLocalFinal(cudaStream_t stream);

    void executeHaloIntermediate(cudaStream_t stream);
    void executeHaloFinal(cudaStream_t stream);
    
private:

    using ChannelActivityMap = std::map<std::string, Interaction::ActivePredicate>;
    
    std::map<CellList*, ChannelActivityMap> cellIntermediateOutputChannels;
    std::map<CellList*, ChannelActivityMap> cellIntermediateInputChannels;
    std::map<CellList*, ChannelActivityMap> cellFinalChannels;
    
    struct InteractionPrototype
    {
        Interaction *interaction;
        ParticleVector *pv1, *pv2;
        CellList *cl1, *cl2;
    };

    std::vector<InteractionPrototype> intermediateInteractions;
    std::vector<InteractionPrototype> finalInteractions;

private:

    void _addChannels(const std::vector<Interaction::InteractionChannel>& src,
                      std::map<std::string, Interaction::ActivePredicate>& dst) const;

    void _executeLocal(std::vector<InteractionPrototype>& interactions, cudaStream_t stream);
    void _executeHalo(std::vector<InteractionPrototype>& interactions, cudaStream_t stream);

};
