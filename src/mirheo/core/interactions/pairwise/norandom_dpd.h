// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/norandom_dpd.h"

namespace mirheo {

/** \brief Interaction to compute dissipative particle dynamics (DPD) forces without the random term.
    Useful for testing.

    This interaction needs the particles positions and velocities.
 */
class PairwiseNoRandomDPDInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseNoRandomDPDInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the DPD forces.
     */
    PairwiseNoRandomDPDInteraction(const MirState *state, const std::string& name, real rc, NoRandomDPDParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseNoRandomDPD pair_;
};

} // namespace mirheo
