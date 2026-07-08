// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/dpd_visco_elastic.h"
#include "stress.h"

#include <mirheo/core/interactions/utils/step_random_gen.h>

#include <optional>

namespace mirheo {

/** \brief Interaction to compute extended dissipative particle dynamics (DPD) forces with visco-elastic component.

    This interaction requires an extra state variable (polymeric chain vector end to end Q) per particle.
    This interaction needs the particles positions, velocities and polymeric chain vectors.
    In addition to forces, the time derivative of the polymeric chain vectors (dQ/dt) is computed.
    This interaction must be used with a special integrator that also evolves the polymeric chain vectors over time.
 */
class PairwiseViscoElasticDPDInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseViscoElasticDPDInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the DPD forces.
        \param [in] stressPeriod The simulation time between two stress computations.
                       If set to `std::nullopt`, disables stress computation.
     */
    PairwiseViscoElasticDPDInteraction(const MirState *state, const std::string& name,
                                       real rc, ViscoElasticDPDParams params,
                                       std::optional<real> stressPeriod=std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getInputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    ViscoElasticDPDParams params_;
    PairwiseViscoElasticDPD pair_;
    std::optional<PairwiseStressWrapper<PairwiseViscoElasticDPD>> pairWithStress_;
    std::optional<StressManager> stressManager_;


    StepRandomGen stepGen_{42424242L};
};

} // namespace mirheo
