#include "restart_helpers.h"

namespace RestartHelpers
{

void copyShiftCoordinates(const DomainInfo &domain, const std::vector<Particle> &parts, LocalParticleVector *local)
{
    local->resize(parts.size(), 0);

    for (int i = 0; i < parts.size(); i++) {
        auto p = parts[i];
        p.r = domain.global2local(p.r);
        local->coosvels[i] = p;
    }
}

} // namespace RestartHelpers
