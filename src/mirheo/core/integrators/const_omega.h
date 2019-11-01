#pragma once

#include "interface.h"

namespace mirheo
{

/**
 * Rotate the particles around #center (defined in global coordinate system)
 * with the angular velocity #omega. All the forces are disregarded.
 *
 * Useful for implementing Taylor-Couette flow (see examples)
 */
class IntegratorConstOmega : public Integrator
{
public:

    IntegratorConstOmega(const MirState *state, std::string name, real3 center, real3 omega);

    ~IntegratorConstOmega();

    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

private:

    real3 center, omega;
};

} // namespace mirheo
