#pragma once

#include <mpi.h>

class ParticleVector;

class InitialConditions
{
public:
	virtual void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream) = 0;

	virtual ~InitialConditions() = default;
};
