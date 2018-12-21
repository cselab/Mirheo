#pragma once

#include "interface.h"

/**
 * Rotate the particles around #center (defined in global coordinate system)
 * with the angular velocity #omega. All the forces are disregarded.
 *
 * Useful for implementing Taylor-Couette flow (see examples)
 */
class IntegratorConstOmega : public Integrator
{
public:

    IntegratorConstOmega(const YmrState *state, std::string name, float3 center, float3 omega);

    ~IntegratorConstOmega();

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

private:

    float3 center, omega;
};
