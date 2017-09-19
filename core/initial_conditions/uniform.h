#pragma once

#include "interface.h"

class UniformIC : public InitialConditions
{
private:
	float density;

public:
	UniformIC(float density) : density(density) {}

	void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream) override;

	~UniformIC() = default;
};

