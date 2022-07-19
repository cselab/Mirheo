// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/sdpd.h"
#include "stress.h"

#include <memory>
#include <optional>

namespace mirheo {

template<class PressureEOS, class DensityKernel>
class PairwiseSDPDInteraction: public BasePairwiseInteraction
{
public:
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
