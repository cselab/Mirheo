#include "factory.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

namespace IntegratorFactory
{

std::shared_ptr<Integrator>
loadIntegrator(const MirState *state, Loader& loader,
               const ConfigObject& config, const std::string& type)
{
    if (type == "IntegratorVV<Forcing_None>")
        return std::make_shared<IntegratorSubStep>(state, loader, config);
    if (type == "IntegratorSubStep")
        return std::make_shared<IntegratorSubStep>(state, loader, config);
    die("Unrecognized or unimplemented integrator: %s", type.c_str());
}

} // namespace IntegratorFactory
} // namespace mirheo
