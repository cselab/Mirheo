// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/mdpd.h"

namespace mirheo {

class PairwiseMDPDInteraction: public BasePairwiseInteraction
{
public:
    PairwiseMDPDInteraction(const MirState *state, const std::string& name, real rc, MDPDParams params);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getInputChannels() const override;

private:
    PairwiseMDPD pair_;
};

} // namespace mirheo
