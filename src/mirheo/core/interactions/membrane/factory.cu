// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"

#include "membrane.h"

#include "force_kernels/common.h"
#include "force_kernels/dihedral/kantor.h"
#include "force_kernels/dihedral/juelicher.h"
#include "force_kernels/triangle/lim.h"
#include "force_kernels/triangle/wlc.h"

#include <mirheo/core/utils/variant_foreach.h>

namespace mirheo
{

std::shared_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, const std::string& name,
                          CommonMembraneParameters commonParams,
                          VarBendingParams varBendingParams, VarShearParams varShearParams,
                          bool stressFree, real initLengthFraction, real growUntil, VarMembraneFilter varFilter)
{
    std::shared_ptr<BaseMembraneInteraction> impl;

    std::visit([&](auto bendingParams, auto shearParams, auto filter)
    {
        using DihedralForce = typename decltype(bendingParams)::DihedralForce;

        if (stressFree)
        {
            using TriangleForce = typename decltype(shearParams)::TriangleForce <StressFreeState::Active>;

            impl = std::make_shared<MembraneInteraction<TriangleForce, DihedralForce, decltype(filter)>>
                (state, name, commonParams, shearParams, bendingParams, initLengthFraction, growUntil, filter);
        }
        else
        {
            using TriangleForce = typename decltype(shearParams)::TriangleForce <StressFreeState::Inactive>;

            impl = std::make_shared<MembraneInteraction<TriangleForce, DihedralForce, decltype(filter)>>
                (state, name, commonParams, shearParams, bendingParams, initLengthFraction, growUntil, filter);
        }
    }, varBendingParams, varShearParams, varFilter);

    return impl;
}

} // namespace mirheo
