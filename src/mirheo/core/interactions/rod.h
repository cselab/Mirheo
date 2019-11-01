#pragma once

#include "interface.h"
#include "rod/kernels/parameters.h"

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

using VarSpinParams = mpark::variant<StatesParametersNone,
                                     StatesSmoothingParameters,
                                     StatesSpinParameters>;

class RodInteraction : public Interaction
{
public:
    RodInteraction(const MirState *state, std::string name,
                   RodParameters params, VarSpinParams spinParams, bool saveEnergies);
    ~RodInteraction();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const override;
};

} // namespace mirheo
