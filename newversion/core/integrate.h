#pragma once

#include "containers.h"

void integrateNoFlow (ParticleVector& pv, const float dt, const float mass, cudaStream_t stream);
void integrateConstDP(ParticleVector& pv, const float dt, const float mass, const float3 extraForce, cudaStream_t stream);
