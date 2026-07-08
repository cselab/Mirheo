// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/dpd.h"
#include "stress.h"

#include <optional>

namespace mirheo {

/** \brief Interaction to compute dissipative particle dynamics (DPD) forces.

    This interaction needs the particles positions and velocities.
 */
class PairwiseDPDInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseDPDInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the DPD forces.
        \param [in] stressPeriod The simulation time between two stress computations.
                       If set to `std::nullopt`, disables stress computation.
     */
    PairwiseDPDInteraction(const MirState *state, const std::string& name,
                           real rc, DPDParams params,
                           std::optional<real> stressPeriod=std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getOutputChannels() const override;

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart(MPI_Comm comm, const std::string& path) override;

private:
    PairwiseDPD pair_;
    std::optional<PairwiseStressWrapper<PairwiseDPD>> pairWithStress_;
    std::optional<StressManager> stressManager_;
};

} // namespace mirheo
