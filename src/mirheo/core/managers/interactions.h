#pragma once

#include <mirheo/core/interactions/interface.h>

#include <map>
#include <string>
#include <vector>

namespace mirheo
{

class LocalParticleVector;

/** \brief Used to manage the execution of interactions.

    One instance of this object represents a stage of interactions (e.g. we need 2 stages for SDPD: densities and forces).
 */
class InteractionManager
{
public:
    InteractionManager() = default;
    ~InteractionManager() = default;

    /// aregister an interaction with the given particle vectors and cell lists
    void add(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    bool empty() const; ///< \return \c true if no interactions were registered

    CellList* getLargestCellList(ParticleVector *pv) const; ///< \return cell list with largest cutoff radius of the given ParticleVector
    real getLargestCutoff() const; ///< \return The largest cut off of all registered cell lists

    std::vector<std::string> getInputChannels(ParticleVector *pv) const;  ///< \return the list of all required channels for this stage
    std::vector<std::string> getOutputChannels(ParticleVector *pv) const; ///< \return the list of all channels that are outputs of this stage

    void clearInput(ParticleVector *pv, cudaStream_t stream); ///< clear input channels of the given ParticleVector
    void clearInputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const;  ///< clear input channels of the given LocalParticleVector

    void clearOutput(ParticleVector *pv, cudaStream_t stream);  ///< clear output channels of the given ParticleVector
    void clearOutputLocalPV(ParticleVector *pv, LocalParticleVector *lpv, cudaStream_t stream) const; ///< clear output channels of the given LocalParticleVector

    void accumulateOutput  (cudaStream_t stream); ///< accumulate all output channels of all registerd ParticleVector objects
    void gatherInputToCells(cudaStream_t stream); ///< gather all the input channels of the registered ParticleVector objects into cell lists

    void executeLocal(cudaStream_t stream); ///< execute the local interactions
    void executeHalo (cudaStream_t stream); ///< execute the halo interactions

    /** \brief check if the output of this stage is compatible with the input of the next

        This function will die if there is a missing output.
     */
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
