#pragma once

#include <mpi.h>
#include <core/domain.h>
#include <cuda_runtime.h>

class ParticleVector;

/**
 * Interface for classes implementing initial conditions
 * ICs are temporary objects, thus not needing name or chkpt/restart
 *
 * exec() member is called by the Simulation right when the ParticleVector
 * is registered
 */
class InitialConditions
{
public:
    virtual void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) = 0;

    virtual ~InitialConditions() = default;
};
