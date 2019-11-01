#include <mirheo/core/datatypes.h>

#include <functional>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector_types.h>

using PositionFilter = std::function<bool(real3)>;

class ParticleVector;

void addUniformParticles(real density, const MPI_Comm& comm, ParticleVector *pv, PositionFilter filterOut, cudaStream_t stream);
