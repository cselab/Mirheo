#pragma once

#include "interface.h"
#include "rod/parameters.h"

#include <extern/variant/include/mpark/variant.hpp>

using VarSpinParams = mpark::variant<StatesParametersNone,
                                     StatesSmoothingParameters,
                                     StatesSpinParameters>;

class InteractionRod : public Interaction
{
public:
    InteractionRod(const YmrState *state, std::string name,
                   RodParameters params, VarSpinParams spinParams, bool saveEnergies);
    virtual ~InteractionRod();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const override;
};
