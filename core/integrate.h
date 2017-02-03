#pragma once

class ParticleVector;

void integrateNoFlow(ParticleVector* pv, const float dt, cudaStream_t stream);
void integrateConstDP(ParticleVector* pv, const float dt, cudaStream_t stream, float3 extraForce);
