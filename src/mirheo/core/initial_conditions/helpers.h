#include <mirheo/core/datatypes.h>

#include <functional>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace mirheo
{

/// \brief Returns `true` if the position is in, `false` otherwise.
using PositionFilter = std::function<bool(real3)>;

class ParticleVector;

/** \brief Create particles uniformly inside a given domain.
    \param [in] numberDensity The target number density of particles to generate.
    \param [in] comm MPI communicator with Cartesian topology.
    \param [in,out] pv ParticleVector that will store the new particles.
    \param [in] filterOut Indicator function that is true inside the considered domain.
    \param [in] stream The stream used to upload data.
 */
void setUniformParticles(real numberDensity, const MPI_Comm& comm, ParticleVector *pv, PositionFilter filterOut, cudaStream_t stream);

} // namespace mirheo
