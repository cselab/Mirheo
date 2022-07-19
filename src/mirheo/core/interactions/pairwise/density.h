// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/density.h"

#include <memory>

namespace mirheo {

template<class DensityKernel>
class PairwiseDensityInteraction: public BasePairwiseInteraction
{
public:
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
