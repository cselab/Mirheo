// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/density.h"

#include <memory>

namespace mirheo {

/** \brief Interaction to compute the density per particle.
    \tparam DensityKernel The blob function used to compute the density.

    This interaction needs particles positions and adds the particles
    contributions to the number density channel.
 */
template<class DensityKernel>
class PairwiseDensityInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseDensityInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the density kernel.
     */
    PairwiseDensityInteraction(const MirState *state, const std::string& name, real rc, DensityParams params);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;


    Stage getStage() const override;

    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    PairwiseDensity<DensityKernel> pair_;
};


/// Factory helper to instantiate all combinations of Density kernels.
std::unique_ptr<BasePairwiseInteraction>
makePairwiseDensityInteraction(const MirState *state, const std::string& name, real rc, DensityParams params);

} // namespace mirheo
