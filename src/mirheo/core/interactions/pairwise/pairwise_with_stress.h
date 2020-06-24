#pragma once

#include "pairwise.h"
#include "kernels/stress_wrapper.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/interactions/interface.h>

#include <map>

namespace mirheo
{

/** \brief Short-range symmetric pairwise interactions with stress output
    \tparam PairwiseKernel The functor that describes the interaction between two particles (interaction kernel).

    This object manages two interaction: one with stress, which is used every stressPeriod time,
    and one with no stress wrapper, that is used the rest of the time.
    This is motivated by the fact that stresses are not needed for the simulation but rather
    for post processing; thus the stresses may not need to be computed at every time step.
 */
template<class PairwiseKernel>
class PairwiseInteractionWithStress : public BasePairwiseInteraction
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // bug in breathe
    /// The parameters corresponding to the interaction kernel.
    using KernelParams = typename PairwiseKernel::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Construct a PairwiseInteractionWithStress object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] rc The cut-off radius of the interaction
        \param [in] stressPeriod The simulation time between two stress computation
        \param [in] pairParams The parameters used to construct the interaction kernel
        \param [in] seed used to initialize random number generator (needed to construct some interaction kernels).
     */
    PairwiseInteractionWithStress(const MirState *state, const std::string& name, real rc, real stressPeriod,
                                  KernelParams pairParams, long seed = 42424242) :
        BasePairwiseInteraction(state, name, rc),
        stressPeriod_(stressPeriod),
        interactionWithoutStress_(state, name, rc, pairParams, seed),
        interactionWithStress_(state, name + "_withStress", rc, pairParams, seed)
    {}

    ~PairwiseInteractionWithStress() = default;

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        interactionWithoutStress_.setPrerequisites(pv1, pv2, cl1, cl2);

        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }

    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const real t = getState()->currentTime;

        if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
        {
            debug("Executing interaction '%s' with stress", getCName());

            interactionWithStress_.local(pv1, pv2, cl1, cl2, stream);
            lastStressTime_ = t;
        }
        else
        {
            interactionWithoutStress_.local(pv1, pv2, cl1, cl2, stream);
        }
    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const real t = getState()->currentTime;

        if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
        {
            debug("Executing interaction '%s' with stress", getCName());

            interactionWithStress_.halo(pv1, pv2, cl1, cl2, stream);
            lastStressTime_ = t;
        }
        else
        {
            interactionWithoutStress_.halo(pv1, pv2, cl1, cl2, stream);
        }
    }

    void setSpecificPair(const std::string& pv1name, const std::string& pv2name, const ParametersWrap::MapParams& mapParams) override
    {
        interactionWithoutStress_.setSpecificPair(pv1name, pv2name, mapParams);
        interactionWithStress_   .setSpecificPair(pv1name, pv2name, mapParams);
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        return interactionWithoutStress_.getInputChannels();
    }

    std::vector<InteractionChannel> getOutputChannels() const override
    {
        auto channels = interactionWithoutStress_.getOutputChannels();

        auto activePredicateStress = [this]()
        {
            const real t = getState()->currentTime;
            return (lastStressTime_+stressPeriod_ <= t) || (lastStressTime_ == t);
        };

        channels.push_back({channel_names::stresses, activePredicateStress});

        return channels;
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        interactionWithoutStress_.checkpoint(comm, path, checkpointId);
        interactionWithStress_   .checkpoint(comm, path, checkpointId);
    }

    void restart(MPI_Comm comm, const std::string& path) override
    {
        interactionWithoutStress_.restart(comm, path);
        interactionWithStress_   .restart(comm, path);
    }

    /// \return A string that describes the type of this object
    static std::string getTypeName()
    {
        return constructTypeName("PairwiseInteractionWithStress", 1,
                                 PairwiseKernel::getTypeName().c_str());
    }

private:
    real stressPeriod_; ///< The stress will be computed every this amount of time
    real lastStressTime_ {-1e6}; ///< to keep track of the last time stress was computed

    PairwiseInteraction<PairwiseKernel> interactionWithoutStress_; ///< The interaction without stress wrapper
    PairwiseInteraction<PairwiseStressWrapper<PairwiseKernel>> interactionWithStress_; ///< The interaction with stress wrapper
};

} // namespace mirheo
