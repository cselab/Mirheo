#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

namespace mirheo
{

class ParticleVector;

/**
 * Interface for classes implementing initial conditions (ICs)
 * ICs are temporary objects, thus not needing name or chekpoint/restart mechanism
 *
 * exec() member is called by the Simulation when the ParticleVector
 * is registered
 */
class InitialConditions
{
public:
    virtual void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) = 0;

    virtual ~InitialConditions() = default;
};

} // namespace mirheo
