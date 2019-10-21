#pragma once

#include <core/pvs/views/rov.h>

namespace RigidOperations
{
enum class ApplyTo { PositionsOnly, PositionsAndVelocities };

void collectRigidForces(const ROVview& view, cudaStream_t stream);

void applyRigidMotion(const ROVview& view, const PinnedBuffer<float4>& initialPositions,
                      ApplyTo action, cudaStream_t stream);

void clearRigidForcesFromMotions(const ROVview& view, cudaStream_t stream);

} // namespace RigidOperations
