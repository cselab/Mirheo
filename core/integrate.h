#pragma once

class ParticleVector;
class RigidObjectVector;

void integrateNoFlow(ParticleVector* pv, const float dt, cudaStream_t stream);
void integrateConstDP(ParticleVector* pv, const float dt, cudaStream_t stream, float3 extraForce);
void integrateRigid(RigidObjectVector* ov, const float dt, cudaStream_t stream, float3 extraForce);
