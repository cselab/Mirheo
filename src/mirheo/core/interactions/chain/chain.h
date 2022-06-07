// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/interactions/interface.h>
#include <optional>

namespace mirheo
{

/** \brief Chain interactions: FENE springs.

    Chain interactions must be used with a ChainVector.
    They are internal forces, meaning that halo() does not compute anything.
 */
class ChainInteraction : public Interaction
{
public:
    /** \brief Construct a \c ChainInteraction
        \param [in] state The global state of the system
        \param [in] name Name of the interaction
        \param [in] ks Spring constant
        \param [in] rmax Maximum extension length
        \param [in] stressPeriod The period (in simulation time) to compute stresses. nullopt for no stress computation.
    */
    ChainInteraction(const MirState *state, const std::string& name, real ks, real rmax,
                     std::optional<real> stressPeriod = std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    bool isSelfObjectInteraction() const final;

    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    bool _isStressTime() const;

private:
    real ks_;
    real rmax2_;

    std::optional<real> stressPeriod_; ///< The stress will be computed every this amount of time
    std::optional<real> lastStressTime_; ///< to keep track of the last time stress was computed
};

} // namespace mirheo
