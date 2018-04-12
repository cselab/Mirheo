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

	void stage1(ParticleVector* pv, float t, cudaStream_t stream) override {};
	void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

	IntegratorConstOmega(std::string name, float dt, float3 center, float3 omega) :
		Integrator(name, dt),
		center(center),	omega(omega)
	{}

	~IntegratorConstOmega() = default;

private:

	float3 center, omega;
};
