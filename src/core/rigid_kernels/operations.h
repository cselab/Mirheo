#pragma once

#include <core/pvs/views/rov.h>

namespace RigidOperations
{
enum class ApplyTo { PositionsOnly, PositionsAndVelocities };

void collectRigidForces(ROVview view, cudaStream_t stream);
void applyRigidMotion  (ROVview view, const PinnedBuffer<float4>& initialPositions,
                        ApplyTo action, cudaStream_t stream);
void clearRigidForces(ROVview view, cudaStream_t stream);

} // namespace RigidOperations
