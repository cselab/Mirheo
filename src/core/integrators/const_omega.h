#pragma once

#include "interface.h"
#include <core/utils/pytypes.h>

/**
 * Rotate the particles around #center (defined in global coordinate system)
 * with the angular velocity #omega. All the forces are disregarded.
 *
 * Useful for implementing Taylor-Couette flow (see examples)
 */
class IntegratorConstOmega : public Integrator
{
public:

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    IntegratorConstOmega(std::string name, float dt, pyfloat3 center, pyfloat3 omega);

    ~IntegratorConstOmega();

private:

    float3 center, omega;
};
