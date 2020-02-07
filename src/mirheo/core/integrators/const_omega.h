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

    IntegratorConstOmega(const MirState *state, const std::string& name, real3 center, real3 omega);
    ~IntegratorConstOmega();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:

    real3 center_, omega_;
};

} // namespace mirheo
