#pragma once

#include "interface.h"

class UniformIC : public InitialConditions
{
private:
	float density;

public:
	UniformIC(float density) : density(density) {}

	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~UniformIC() = default;
};

