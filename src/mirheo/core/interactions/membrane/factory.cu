#include "factory.h"

#include "membrane.h"

#include "force_kernels/common.h"
#include "force_kernels/dihedral/kantor.h"
#include "force_kernels/dihedral/juelicher.h"
#include "force_kernels/triangle/lim.h"
#include "force_kernels/triangle/wlc.h"

namespace mirheo
{

std::unique_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, const std::string& name,
                          CommonMembraneParameters commonParams,
                          VarBendingParams varBendingParams, VarShearParams varShearParams,
                          bool stressFree, real growUntil, VarMembraneFilter varFilter)
{
    std::unique_ptr<BaseMembraneInteraction> impl;

    mpark::visit([&](auto bendingParams, auto shearParams, auto filter)
    {                     
        using FilterType    = decltype(filter);
        using DihedralForce = typename decltype(bendingParams)::DihedralForce;
        
        if (stressFree)
        {
            using TriangleForce = typename decltype(shearParams)::TriangleForce <StressFreeState::Active>;
            
            impl = std::make_unique<MembraneInteraction<TriangleForce, DihedralForce, FilterType>>
                (state, name, commonParams, shearParams, bendingParams, growUntil, filter);
        }
        else
        {
            using TriangleForce = typename decltype(shearParams)::TriangleForce <StressFreeState::Inactive>;
            
            impl = std::make_unique<MembraneInteraction<TriangleForce, DihedralForce, FilterType>>
                (state, name, commonParams, shearParams, bendingParams, growUntil, filter);
        }
    }, varBendingParams, varShearParams, varFilter);

    return std::move(impl);
}

} // namespace mirheo
