// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"

#include "triplewise.h"

#include "kernels/sw.h"
#include "kernels/dummy.h"

namespace mirheo
{

std::shared_ptr<BaseTriplewiseInteraction>
createInteractionTriplewise(const MirState *state, const std::string& name, real rc, const VarTriplewiseParams& varParams)
{
    // NOTE: This is a simplified version of the force. We assume that stresses are not needed.
    return std::visit([&](const auto& params) -> std::shared_ptr<BaseTriplewiseInteraction>
    {
        using Kernel = typename std::remove_reference_t<decltype(params)>::KernelType;
        return std::make_shared<TriplewiseInteraction<Kernel>>(state, name, rc, params);
    }, varParams);
}

} // namespace mirheo
