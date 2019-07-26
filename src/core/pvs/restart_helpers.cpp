#include "restart_helpers.h"

#include <core/utils/cuda_common.h>

namespace RestartHelpers
{

std::tuple<std::vector<float3>,
           std::vector<float3>,
           std::vector<int64_t>>
splitAndShiftPosVel(const DomainInfo &domain,
                    const PinnedBuffer<float4>& pos4,
                    const PinnedBuffer<float4>& vel4)
{
    auto n = pos4.size();
    std::vector<float3> pos(n), vel(n);
    std::vector<int64_t> ids(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto p = Particle(pos4[i], vel4[i]);
        pos[i] = domain.local2global(p.r);
        vel[i] = p.u;
        ids[i] = p.getId();
    }
    return {pos, vel, ids};
}


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
