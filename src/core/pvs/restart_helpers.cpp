#include "restart_helpers.h"

namespace RestartHelpers
{

void copyShiftCoordinates(const DomainInfo &domain, const std::vector<float4>& pos, const std::vector<float4>& vel,
                          LocalParticleVector *local)
{
    local->resize(pos.size(), 0);

    float4 *positions  = local->positions() .data();
    float4 *velocities = local->velocities().data();
    
    for (int i = 0; i < pos.size(); i++) {
        auto p = Particle(pos[i], vel[i]);
        p.r = domain.global2local(p.r);
        positions [i] = p.r2Float4();
        velocities[i] = p.u2Float4();
    }
}

} // namespace RestartHelpers
