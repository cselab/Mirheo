#pragma once

#include <mpi.h>
#include <core/domain.h>

class ParticleVector;

/**
 * Interface for classes implementing initial conditions
 *
 * exec() member is called by the Simulation right when the ParticleVector
 * is registered
 */
class InitialConditions
{
public:
	virtual void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) = 0;

	virtual ~InitialConditions();
};
