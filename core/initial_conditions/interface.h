#pragma once

#include <mpi.h>
#include <core/domain.h>

class ParticleVector;

/**
 * Interface for classes implementing initial conditions
 *
 * \c exec member is called by the \c Simulation right when the \c ParticleVector
 * is registered
 */
class InitialConditions
{
public:
	virtual void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) = 0;

	virtual ~InitialConditions() = default;
};
