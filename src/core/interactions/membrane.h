#pragma once

#include "interface.h"
#include "membrane/parameters.h"

#include <extern/variant/include/mpark/variant.hpp>

#include <memory>

using VarBendingParams = mpark::variant<KantorBendingParameters, JuelicherBendingParameters>;
using VarShearParams   = mpark::variant<WLCParameters, LimParameters>;

/**
 * membrane interactions
 * forces depend on the passed variant parameters
 */
class InteractionMembrane : public Interaction
{
public:

    InteractionMembrane(const YmrState *state, std::string name, CommonMembraneParameters commonParams,
                        VarBendingParams bendingParams, VarShearParams shearParams, bool stressFree, float growUntil);
    ~InteractionMembrane();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

protected:

    /**
     * compute quantities used by the force kernels.
     * this is called before every force kernel (see implementation of @ref local)
     * default: compute area and volume of each cell
     */
    virtual void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream);
};
