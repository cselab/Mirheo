#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

namespace mirheo
{

class ParticleVector;

/** \brief Initializer for objects in group PVs.

    ICs are temporary objects and do not need name or checkpoint/restart mechanism.
    The exec() member function is called by the Simulation when the ParticleVector
    is registered.
 */
class InitialConditions
{
public:
    virtual ~InitialConditions() = default;

    /** \brief Initialize a given ParticleVector.
        \param [in] comm A Cartesian MPI communicator from the simulation tasks
        \param [in,out] pv The resulting ParticleVector to be initialized (on chip data) 
        \param [in] stream cuda stream
     */
    virtual void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) = 0;
};

} // namespace mirheo
