#pragma once

#include <mirheo/core/datatypes.h>

#include <mpi.h>
#include <cuda_runtime.h>

namespace mirheo
{

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
    virtual void exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream) = 0;

    virtual ~InitialConditions() = default;
};

} // namespace mirheo
