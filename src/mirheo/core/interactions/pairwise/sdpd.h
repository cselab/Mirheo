// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/sdpd.h"
#include "stress.h"

#include <memory>
#include <optional>

namespace mirheo {

/** \brief Interaction to compute smooth dissipative particle dynamics (SDPD) forces.
    \tparam PressureEOS The kernel used to compute the equation of state.
    \tparam DensityKernel The kernel used to compute the number densities.

    This interaction needs the particles positions velocities and densities.
 */
template<class PressureEOS, class DensityKernel>
class PairwiseSDPDInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseSDPDInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the DPD forces.
        \param [in] stressPeriod The simulation time between two stress computations.
                       If set to `std::nullopt`, disables stress computation.
     */
    PairwiseSDPDInteraction(const MirState *state, const std::string& name,
                            real rc, SDPDParams params,
                            std::optional<real> stressPeriod=std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getInputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    PairwiseSDPD<PressureEOS, DensityKernel> pair_;
    std::optional<PairwiseStressWrapper<PairwiseSDPD<PressureEOS, DensityKernel>>> pairWithStress_;
    std::optional<StressManager> stressManager_;
};


/// Factory helper to instantiate all combinations of PressureEOS and DensityKernel.
std::unique_ptr<BasePairwiseInteraction>
makePairwiseSDPDInteraction(const MirState *state, const std::string& name, real rc,
                            SDPDParams params, std::optional<real> stressPeriod);

} // namespace mirheo
