// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

namespace integrator_factory
{

std::shared_ptr<Integrator>
loadIntegrator(const MirState *state, Loader& loader, const ConfigObject& config)
{
    const std::string& type = config["__type"];
    if (type == "IntegratorMinimize")
        return std::make_shared<IntegratorMinimize>(state, loader, config);
    if (type == "IntegratorVV<ForcingTermNone>")
        return std::make_shared<IntegratorVV<ForcingTermNone>>(state, loader, config);
    if (type == "IntegratorSubStep")
        return std::make_shared<IntegratorSubStep>(state, loader, config);
    die("Unrecognized or unimplemented integrator: %s", type.c_str());
}

} // namespace integrator_factory
} // namespace mirheo
