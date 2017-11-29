#pragma once

#include <mpi.h>
#include <core/domain.h>

class ParticleVector;

class InitialConditions
{
public:
	virtual void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) = 0;

	virtual ~InitialConditions() = default;
};
