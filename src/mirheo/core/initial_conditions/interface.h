#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

namespace mirheo
{

class ParticleVector;

/**
   \defgroup ICs Initial Conditions
 */

/** \brief Initializer for objects in group PVs
    \ingroup ICs

    ICs temporary objects and do not needi name or chekpoint/restart mechanism
    exec() member is called by the Simulation when the ParticleVector
    is registered
 */
class InitialConditions
{
public:
    virtual ~InitialConditions() = default;

    /** \brief Initialize a given ParticleVector
        \param [in] comm A carthesian MPI communicator from the simulation tasks
        \param [out] pv The resulting ParyicleVector to be initialized (on chip data) 
        \param [in] stream cuda stream
     */
    virtual void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) = 0;
};

} // namespace mirheo
