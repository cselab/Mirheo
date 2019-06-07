#include "restart_helpers.h"

#include <core/utils/cuda_common.h>

namespace RestartHelpers
{

void copyShiftCoordinates(const DomainInfo &domain, const std::vector<float4>& pos, const std::vector<float4>& vel,
                          LocalParticleVector *local)
{
    auto& positions  = local->positions();
    auto& velocities = local->velocities();

    positions .resize(pos.size(), defaultStream);
    velocities.resize(vel.size(), defaultStream);
    
    for (int i = 0; i < pos.size(); i++) {
        auto p = Particle(pos[i], vel[i]);
        p.r = domain.global2local(p.r);
        positions [i] = p.r2Float4();
        velocities[i] = p.u2Float4();
    }
}

} // namespace RestartHelpers
